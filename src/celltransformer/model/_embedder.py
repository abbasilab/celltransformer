"""Cell embedding: expression MLP + cell-type embedding → model_dim."""

from typing import Optional, Literal

import torch
from torch import nn
from jaxtyping import Float, Int64

from celltransformer.model._layers import DiagonalNormalization


class CellEmbedder(nn.Module):
    def __init__(self,
                 expression_dim: int,
                 cell_type_cardinality: int,
                 output_dim: int,
                 concat_mode: Optional[Literal["cat", "add"]],
                 use_diagonal_norm: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}

        if expression_dim <= 0 or cell_type_cardinality <= 0:
            raise ValueError("Dimensions must be positive integers.")

        if output_dim % 2 != 0 and concat_mode == "cat":
            raise ValueError("Output dimension must be even when using concatenation mode.")

        if concat_mode not in ["cat", "add"]:
            raise ValueError("concat_mode must be either 'cat' or 'add'.")

        self.expression_dim = expression_dim
        self.output_dim = output_dim
        self.cell_type_cardinality = int((cell_type_cardinality + 63) // 64 * 64)  # round to nearest 64
        self.embed_dim = self.output_dim // 2 if concat_mode == "cat" else self.output_dim
        self.concat_mode = concat_mode

        self.cell_type_subnet = nn.Embedding(self.cell_type_cardinality, self.embed_dim,
                                             **factory_kwargs
                                             )

        # applies a per-feature linear shift/scale (eg matrix prod + bias)
        if use_diagonal_norm:
            self.expression_normalization = DiagonalNormalization(self.expression_dim, **factory_kwargs)
        else:
            self.expression_normalization = nn.Identity()

        self.expression_subnet = nn.Sequential(
            self.expression_normalization,
            nn.Linear(self.expression_dim, self.output_dim, **factory_kwargs),
            nn.LayerNorm(self.output_dim, **factory_kwargs),
            nn.GELU(),
            nn.Linear(self.output_dim, self.embed_dim, **factory_kwargs),
            nn.LayerNorm(self.embed_dim, **factory_kwargs),
        )

    def forward(self, expression_mat: Float[torch.Tensor, "num_cells n_genes"],
                      cell_type_annot: Int64[torch.Tensor, "num_cells"]):
        expression_embedding = self.expression_subnet(expression_mat)
        cell_type_embedding = self.cell_type_subnet(cell_type_annot)

        if self.concat_mode == "cat":
            combined_embedding = torch.cat([expression_embedding, cell_type_embedding], dim=-1)
        else:
            combined_embedding = expression_embedding + cell_type_embedding

        return combined_embedding
