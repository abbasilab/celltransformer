from typing import Optional

import torch
from torch import nn

from .blocks import AttnPool, ZINBProj
from .factory import get_projection_layers, set_up_transformer_layers


class CellTransformer(nn.Module):
    def __init__(
        self,
        encoder_embedding_dim: int,
        encoder_num_heads: int,
        encoder_depth: int,
        decoder_embedding_dim: int,
        decoder_num_heads: int,
        decoder_depth: int,
        attn_pool_heads: Optional[int] = 8,
        cell_cardinality: Optional[int] = 1024,
        put_device: Optional[str] = "cuda",
        eps: float = 1e-15,
        n_genes: Optional[int] = 500,
        xformer_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = False,
        zero_attn: Optional[bool] = True,
    ):
        """An encoder-decoder model with attention pooling prior to decoder.

        Parameters
        ----------
        encoder_embedding_dim : int
        encoder_num_heads : int
        encoder_depth : int
        decoder_embedding_dim : int
        decoder_num_heads : int
        decoder_depth : int
        attn_pool_heads : Optional[int], optional
            Number attention pool heads, by default 8
        cell_cardinality : Optional[int], optional
            Cardinality/number of different cell types for embedding, by default 1024
        put_device : Optional[str], optional
            For later referencing where to put input tensors, by default 'cuda'
        eps : float, optional
            Stability additional constant for NB params, by default 1e-15
        n_genes : Optional[int], optional
            Number genes, by default 500
        xformer_dropout : Optional[float], optional
            Dropout %, by default 0.0
        bias : Optional[bool], optional
            Use bias or not, by default True
        zero_attn : Optional[bool], optional
            Enable zero attention (zeros appended to k/v seqs to allow for "attending to nothing"), by default True
        """
        super().__init__()

        self.eps = eps

        _feature_dim = encoder_embedding_dim // 2

        self.cls_token = nn.Parameter(torch.zeros(1, encoder_embedding_dim))

        self.expression_projection = get_projection_layers(n_genes, _feature_dim)
        self.encoder_cell_embed = nn.Embedding(cell_cardinality, _feature_dim)

        self.pooling_token = nn.Parameter(torch.randn(1, encoder_embedding_dim))

        self.proj_norm = nn.LayerNorm(encoder_embedding_dim)

        self.decoder_cell_embed = nn.Embedding(cell_cardinality, decoder_embedding_dim)
        self.attn_pool = AttnPool(encoder_embedding_dim, attn_pool_heads,
                                  bias=True, zero_attn=True)
        # no bias causes instability in training

        self.encoder = set_up_transformer_layers(
            encoder_embedding_dim,
            encoder_num_heads,
            encoder_depth,
            xformer_dropout,
            bias,
            zero_attn,
        )
        #self.encoder = torch.compile(self.encoder)

        self.decoder = set_up_transformer_layers(
            decoder_embedding_dim,
            decoder_num_heads,
            decoder_depth,
            xformer_dropout,
            bias,
            zero_attn,
        )
        #self.decoder = torch.compile(self.decoder)

        self.zinb_proj = ZINBProj(
            embed_dim=decoder_embedding_dim, n_genes=n_genes, eps=self.eps
        )

        nn.init.normal_(self.cls_token, std=0.1) #std=0.02)

        self.put_device = put_device

    def forward(self, data_dict: dict):
        bs = data_dict["bs"]

        cells = data_dict["observed_cell_type"].to(self.put_device, non_blocking=False)
        expression = data_dict["observed_expression"].to(
            self.put_device, non_blocking=False
        )

        num_hidden = len(
            data_dict["masked_cell_type"]
        )  # will in practice be == bs but in case later want to train on multiple cells
        hidden_expression = data_dict["masked_expression"].to(
            self.put_device, non_blocking=False, dtype=torch.float32
        )
        hidden_cells = data_dict["masked_cell_type"].to(
            self.put_device, non_blocking=False
        )

        pooling_mask = data_dict["pooling_mask"].to(self.put_device, non_blocking=False)
        decoder_mask = data_dict["decoder_mask"].to(self.put_device, non_blocking=False)
        encoder_mask = data_dict["encoder_mask"].to(self.put_device, non_blocking=False)

        cls_tokens = self.cls_token.repeat_interleave(bs, dim=0)

        cells_embed = self.encoder_cell_embed(cells)

        expression_embed = self.expression_projection(expression)
        cells_embed = torch.cat((cells_embed, expression_embed), dim=1)
        cells_embed = torch.cat((cells_embed, cls_tokens), dim=0)

        cells_embed = self.proj_norm(cells_embed)

        cells_embed = self.encoder(cells_embed, mask=encoder_mask)

        attn_pool = self.attn_pool(cells_embed, pooling_mask, bs)

        decoding_queries = self.decoder_cell_embed(hidden_cells)

        cells_embed = torch.cat((attn_pool, decoding_queries), dim=0)

        cells_embed = self.decoder(cells_embed, mask=decoder_mask)
        ref_cell_embed = cells_embed[-bs:]

        with torch.autocast(dtype=torch.float32, device_type=self.put_device):
            # sometimes bfloat doesn't work here for lgamma backward (in zinb, otherwise will error)
            zinb_params = self.zinb_proj(ref_cell_embed)

        cls_toks = cells_embed[-(num_hidden + bs) : -num_hidden]

        return dict(
            zinb_params=zinb_params,
            neighborhood_repr=cls_toks,
            hidden_expression=hidden_expression,
        )
