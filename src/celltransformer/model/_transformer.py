"""Top-level SpatialCellTransformer model."""

from typing import Optional, Literal

import torch
from torch import nn

from celltransformer.model._embedder import CellEmbedder
from celltransformer.model._layers import TransformerEncoder
from celltransformer.model._pool import AttentionPool
from celltransformer.model._projection import ZINBProj


class SpatialCellTransformer(nn.Module):
    def __init__(self,
                 expression_dim: int,
                 cell_type_cardinality: int,
                 model_dim: int,
                 encoder_depth: int,
                 decoder_depth: int,
                 num_heads: int,
                 ffn_dim: int,
                 concat_mode: Optional[Literal["cat", "add"]],
                 use_zi: bool = True,
                 use_diagonal_norm: bool = True,
                 nb_proj_eps: float = 1e-4,
                 dropout: float = 0.1,
                 multiple_of: int = 256,
                 norm_eps: float = 1e-5,
                 init_std_attn_pool: float = 0.02,
                 compile: bool = False,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 ):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.nb_proj = ZINBProj(
            embed_dim=model_dim,
            n_genes=expression_dim,
            eps=nb_proj_eps,
            zero_inflated=use_zi,
            **factory_kwargs
        )

        self.embedder = CellEmbedder(
            expression_dim=expression_dim,
            cell_type_cardinality=cell_type_cardinality,
            output_dim=model_dim,
            concat_mode=concat_mode,
            use_diagonal_norm=use_diagonal_norm,
            **factory_kwargs,
        )

        # Encoder receives NJTs — only FFN is safe to compile per-submodule.
        # MHA can't be compiled with NJTs (PyTorch NestedIntNode guard issue).
        self.encoder = nn.Sequential(*[
            TransformerEncoder(
                d_model=model_dim,
                nheads=num_heads,
                dim_feedforward=ffn_dim,
                multiple_of=multiple_of,
                dropout=dropout,
                norm_eps=norm_eps,
                compile=compile,
                **factory_kwargs
            )
            for _ in range(encoder_depth)
        ])

        # AttentionPool receives NJTs — left uncompiled.
        self.attn_pool = AttentionPool(
            input_dim=model_dim,
            output_dim=model_dim,
            nheads=num_heads,
            dropout_path=dropout,
            init_std=init_std_attn_pool,
            **factory_kwargs
        )

        self.decoder_cell_type_subnet = nn.Embedding(cell_type_cardinality, model_dim, **factory_kwargs)

        # Decoder receives regular 3D tensors (bs, 2, dim) — safe to compile
        # as whole modules including MHA.
        self.decoder = nn.Sequential(*[
            TransformerEncoder(
                d_model=model_dim,
                nheads=num_heads,
                dim_feedforward=ffn_dim,
                multiple_of=multiple_of,
                dropout=dropout,
                norm_eps=norm_eps,
                **factory_kwargs
            )
            for _ in range(decoder_depth)
        ])

        if compile:
            self.embedder = torch.compile(self.embedder, dynamic=True)
            self.decoder = torch.compile(self.decoder, dynamic=True)
            self.nb_proj = torch.compile(self.nb_proj, dynamic=True)

    def forward(self, expression_mat: torch.Tensor,
                cell_type_annot_obs: torch.Tensor,
                lengths: torch.Tensor,
                decoder_cell_type_annot: torch.Tensor,
                ):
        if (lengths == 0).any():
            raise ValueError(
                "Zero-length neighborhoods in batch. AttentionPool requires at least "
                "one neighbor per sample (softmax over empty set produces NaN). "
                "Filter out zero-length entries before calling forward()."
            )

        x = self.embedder(expression_mat, cell_type_annot_obs)  # could be dense
        x = torch.nested.nested_tensor_from_jagged(x, lengths=lengths)
        x = self.encoder(x)

        spatial_context = self.attn_pool(x)

        decoder_embeds = self.decoder_cell_type_subnet(decoder_cell_type_annot)  # (bs, output_dim)
        decoder_tokens = torch.cat([spatial_context.values().unsqueeze(1), decoder_embeds.unsqueeze(1)], dim=1)  # (bs, 2, output_dim)
        decoder_tokens = self.decoder(decoder_tokens)

        sc_prediction_toks = decoder_tokens[:, 1, :]  # (bs, output_dim)
        nb_params = self.nb_proj(sc_prediction_toks)

        return dict(nb_params=nb_params,
                    spatial_context=spatial_context,
                    decoder_tokens=decoder_tokens,
                    )
