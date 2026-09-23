"""Per-feature normalization and pre-norm transformer encoder layer."""

import torch
from torch import nn
from jaxtyping import Float

from celltransformer.model._attention import MultiHeadAttention
from celltransformer.model._ffn import PackedSwiGLUFFN


class DiagonalNormalization(nn.Module):
    def __init__(self, in_dim: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.scale = nn.Parameter(torch.ones(in_dim, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(in_dim, **factory_kwargs))

    def forward(self, x: Float[torch.Tensor, "n_cells in_dim"]):
        # equiv to diagonal feature scaling plus bias
        return x * self.scale + self.bias


class TransformerEncoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 nheads: int,
                 dim_feedforward: int,
                 multiple_of: int = 256,
                 dropout: float = 0.0,
                 bias: bool = True,
                 norm_eps: float = 1e-5,
                 compile: bool = False,
                 device=None,
                 dtype=None,
                 ):

        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.self_attn = MultiHeadAttention(
            E_q=d_model,
            E_k=d_model,
            E_v=d_model,
            E_total=d_model,
            nheads=nheads,
            dropout=dropout,
            bias=bias,
            **factory_kwargs,
        )

        self.ffn = PackedSwiGLUFFN(
            dim=d_model,
            hidden_dim=dim_feedforward,
            multiple_of=multiple_of,
            **factory_kwargs,
        )

        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if compile:
            self.ffn = torch.compile(self.ffn, dynamic=True)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None,):
        x_norm = self.norm1(x)

        attn_output = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=attn_mask,
        )

        x = x + self.dropout1(attn_output)

        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout2(ffn_output)

        return x
