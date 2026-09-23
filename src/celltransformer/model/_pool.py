"""Attention pooling: variable-length neighborhood → single context vector."""

from typing import Optional

import torch
from torch import nn

from celltransformer.model._attention import MultiHeadAttention


class AttentionPool(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, nheads: int = 8,
                 dropout_path: float = 0.0,
                 init_std: float = 0.02,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 ):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nheads = nheads

        self.attention = MultiHeadAttention(
            E_q=output_dim,
            E_k=input_dim,
            E_v=input_dim,
            E_total=input_dim,
            nheads=nheads,
            dropout=0.0,
            **factory_kwargs
        )

        self.query_proj = nn.Parameter(torch.zeros(1, output_dim, **factory_kwargs))
        nn.init.normal_(self.query_proj, std=init_std)

        self.norm_v = nn.LayerNorm(input_dim, **factory_kwargs)
        self.norm_k = nn.LayerNorm(input_dim, **factory_kwargs)
        self.norm_q = nn.LayerNorm(output_dim, **factory_kwargs)

        self.attn_dropout = nn.Dropout(dropout_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        query = self.query_proj.expand(batch_size, -1)

        lengths = torch.ones(batch_size, dtype=torch.long, device=x.device)

        query_njt = torch.nested.nested_tensor_from_jagged(query, lengths=lengths)
        key = self.norm_k(x)
        value = self.norm_v(x)

        attn_output = self.attention(query_njt, key, value)

        attn_output = self.attn_dropout(attn_output)
        return attn_output
