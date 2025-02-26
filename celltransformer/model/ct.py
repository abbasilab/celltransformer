from typing import Dict

import numpy as np
import torch
from torch import nn

from .blocks import AttnPool


class CellLocationTransformer(nn.Module):
    def __init__(
        self,
        encoder_embedding_dim: int,
        encoder_num_heads: int,
        encoder_depth: int,
        decoder_embedding_dim: int,
        decoder_num_heads: int,
        decoder_depth: int,
        cell_cardinality: int = 307,
        device: str = "cuda",
        eps: float = 1e-9,
        attn_pool: bool = False,
        attn_pool_heads: int = 8,
        actvn_last: bool = False,
        num_output_genes: int = 500,
    ):
        """An encoder-decoder model with attention pooling prior to decoder. 

        Args:
            encoder_embedding_dim (int): embedding dim 
            encoder_num_heads (int): 
            encoder_depth (int): 
            decoder_embedding_dim (int): 
            decoder_num_heads (int): 
            decoder_depth (int): 
            cell_cardinality (int, optional): Number of cell types. Defaults to 307.
            device (str, optional): Device for transfering tensors to in `forward`. Defaults to "cuda".
            eps (float, optional): Floor for nonnegative parameters of negative binomial regression layers. Defaults to 1e-9.
            attn_pool (bool, optional): Whether to use attention pooling or not. Defaults to False.
            attn_pool_heads (int, optional): Number of attention pool heads. Defaults to 8.
            num_output_genes (int, optional): Number of genes to regress. Defaults to 500.
        """
        super().__init__()

        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.device = device

        self.eps = eps
        self.cls_token = nn.Parameter(torch.zeros(1, encoder_embedding_dim))

        _feature_dim = encoder_embedding_dim // 2

        self.expression_projection = nn.Sequential(
            nn.Linear(num_output_genes, _feature_dim),
            nn.LayerNorm(_feature_dim),
            nn.GELU(),
            nn.Linear(_feature_dim, _feature_dim),
            nn.LayerNorm(_feature_dim),
            nn.GELU() if actvn_last else nn.Identity(),
        )

        self.cell_embedding = nn.Embedding(cell_cardinality, _feature_dim)
        self.decoding_cell_embedding = nn.Embedding(
            cell_cardinality, encoder_embedding_dim
        )

        self.pooling_token = nn.Parameter(torch.randn(1, encoder_embedding_dim))

        self.norm = nn.LayerNorm(encoder_embedding_dim)
        if attn_pool is True:
            self.attn_pool = AttnPool(
                encoder_embedding_dim, attn_pool_heads, bias=True, zero_attn=True
            )
        else:
            self.attn_pool = None

        if encoder_depth != 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=encoder_embedding_dim,
                nhead=encoder_num_heads,
                activation="gelu",
                dropout=0.0,
            )

            encoder_layer.self_attn = nn.MultiheadAttention(
                encoder_embedding_dim,
                num_heads=encoder_num_heads,
                dropout=0.0,
                add_zero_attn=True,
            )

            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=decoder_depth,
                enable_nested_tensor=False,
            )
        else:
            self.encoder = None

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embedding_dim,
            nhead=decoder_num_heads,
            activation="gelu",
            dropout=0.0,
        )

        decoder_layer.self_attn = nn.MultiheadAttention(
            encoder_embedding_dim,
            num_heads=decoder_num_heads,
            dropout=0.0,
            add_zero_attn=True,
        )

        self.decoder = nn.TransformerEncoder(
            encoder_layer=decoder_layer,
            num_layers=decoder_depth,
            enable_nested_tensor=False,
            # num_layers=8,
        )

        self.mu = nn.Linear(encoder_embedding_dim, num_output_genes)  # mean
        self.theta = nn.Linear(
            encoder_embedding_dim, num_output_genes
        )  # inv. dispersion
        self.scale = nn.Linear(encoder_embedding_dim, num_output_genes)  # avg. expr.
        self.gate_logit = nn.Linear(encoder_embedding_dim, num_output_genes)  # zi

        self.pos_encoder = nn.Sequential(
            # nn.LayerNorm(3),
            nn.Linear(3, encoder_embedding_dim),
            nn.LayerNorm(encoder_embedding_dim),
            nn.GELU(),
            nn.Linear(encoder_embedding_dim, encoder_embedding_dim),
            nn.LayerNorm(encoder_embedding_dim),
            nn.GELU() if fancy_encoding else nn.Identity(),
        )

        self._init_weights()

    def set_device(self, device: str):
        self.device = device
        self.to(device)

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)

        # see: unsure if it has been fixed as of 2023-12-04 https://github.com/pytorch/pytorch/issues/72253
        # for param_name, param in self.named_parameters():
        #     if (
        #         'decoder' in param_name
        #         and ('linear' in param_name or 'proj' in param_name)
        #         and 'bias' not in param_name
        #     ):
        #         nn.init.kaiming_normal_(param)

        #     if (
        #         'encoder' in param_name
        #         and ('linear' in param_name or 'proj' in param_name)
        #         and 'bias' not in param_name
        #     ):
        #         nn.init.kaiming_normal_(param)

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        only_pos_encoder: bool = False,
    ):
        bs = data_dict["bs"]
        cells = data_dict["observed_cells"].to(self.device)
        center_indices = data_dict["center_indices"].to(self.device, non_blocking=False)
        attn_mask = data_dict["attn_mask"].to(self.device, non_blocking=True)
        # repeat_lengths = data_dict["hidden_sequence_lengths"].to(
        #     self.device, non_blocking=True
        # )
        hidden_cells = data_dict["hidden_cells"].to(self.device, non_blocking=False)
        hidden_expression = data_dict["hidden_expression"].to(
            self.device, non_blocking=False
        )
        expression = data_dict["observed_expression"].to(
            self.device, non_blocking=False
        )
        # pooling_mask = data_dict["pooling_mask"].to(self.device, non_blocking=True)
        full_mask = data_dict["full_mask"].to(self.device, non_blocking=False)
        # position_coords = data_dict["pos_embed"].to(self.device, non_blocking=True)
        if self.attn_pool is not None:
            post_pool_mask = data_dict["post_pooling_mask"].to(
                self.device, non_blocking=True
            )
            pooling_mask = data_dict["pooling_mask"].to(self.device, non_blocking=True)

        num_tokens_encoder = attn_mask.shape[0]
        num_obs = cells.shape[0]
        num_hidden = hidden_cells.shape[0]

        # pos_tokens = self.pos_encoder(center_indices)
        # pos_tokens = self.pos_encoder(position_coords)
        pos_tokens = self.cls_token.repeat_interleave(bs, dim=0)

        if len(expression) > 0:
            cells_embed = self.cell_embedding(cells)
            # expression_embed = self.expression_projection(expression[:, :500])
            expression_embed = self.expression_projection(expression)
            cells_embed = torch.cat((cells_embed, expression_embed), dim=1)
            cells_embed = torch.cat((cells_embed, pos_tokens), dim=0)
        else:
            cells_embed = pos_tokens

        cells_embed = self.norm(cells_embed)
        # celss_embed = torch.cat((cells_embed, decoding_queries))

        # if only_pos_encoder:
        #    return dict(pos_tokens=encoded[-bs:, :])

        # pos_tokens = encoded[-bs:]
        # pos_tokens = self.attn_pool(encoded, self.pooling_token, pooling_mask)
        # expanded = pos_tokens.repeat_interleave(repeat_lengths, dim=0)
        # logits = self.cell_clf(expanded)

        if self.attn_pool is None:
            # print('attn pool is none')
            decoding_queries = self.decoding_cell_embedding(hidden_cells)

            if self.encoder is not None:
                #    print('encoder is not none')
                cells_embed = self.encoder(cells_embed, mask=attn_mask)
            #                cells_embed = torch.cat()

            all_indices = list(range(full_mask.shape[0]))
            obs_indices = all_indices[:num_obs]
            hidden_indices = all_indices[-num_hidden:]
            # print(full_mask.shape[0], num_hidden, 'alex look here bro')

            # pos_indices = all_indices[num_obs:-num_hidden]
            pos_indices = all_indices[num_obs : num_obs + bs]
            indices_to_use = np.array(obs_indices + hidden_indices)
            early_pos = cells_embed[pos_indices].clone().detach()

            cells_embed = torch.cat((cells_embed, decoding_queries), dim=0)
            cells_embed = self.decoder(cells_embed, mask=full_mask)

            all_cells_embed = cells_embed[indices_to_use]

        # logits = self.gex_decoder(cells_embed[num_tokens_encoder:])

        # observed_expr = self.gex_decoder(cells_embed[:num_cells])
        # logits = self.gex_decoder(cells_embed[num_tokens_encoder:])
        else:
            cells_embed = self.encoder(cells_embed, mask=attn_mask)
            attn_pooled = self.attn_pool(cells_embed, pooling_mask, bs)
            decoding_queries = self.decoding_cell_embedding(hidden_cells)
            cells_embed = torch.cat((attn_pooled, decoding_queries), dim=0)

            cells_embed = self.decoder(cells_embed, mask=post_pool_mask)

            all_indices = list(range(cells_embed.shape[0]))
            obs_indices = None
            # hidden_indices = all_indices[-num_hidden:]
            hidden_indices = all_indices
            early_pos = cells_embed[:bs].clone().detach()

            pos_indices = all_indices[:bs]
            indices_to_use = hidden_indices

            all_cells_embed = cells_embed[-bs:]

        pos_tokens = cells_embed[pos_indices]

        mu = self.mu(all_cells_embed).exp()
        theta = self.theta(all_cells_embed).exp()

        gate = self.gate_logit(all_cells_embed)
        scale = self.scale(all_cells_embed).exp()

        all_expression = torch.cat((expression, hidden_expression))

        return dict(
            zinb_params=dict(mu=mu, scale=scale, theta=theta, zi_logits=gate),
            mu=mu,
            theta=theta,
            gate=gate,
            scale=scale,
            early_pos=early_pos,
            # embeddings=encoded[:bs],
            decodings=cells_embed,
            hidden_cells=hidden_cells,
            pos_tokens=pos_tokens,
            hidden_expression=hidden_expression,
            all_expression=all_expression,
            hidden_indices=np.array(hidden_indices),
            # all_expression=torch.cat((expression, hidden_expression))
        )
