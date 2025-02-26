from typing import Dict, Optional

import torch
from jaxtyping import Float
from torch import nn


class ZINBProj(nn.Module):

    def __init__(self, embed_dim: int, n_genes: int, eps: float):
        """A utility block to regress the parameters of a (zero-inflated)
        negative binomial distribution from an embedding. 

        Args:
            embed_dim (int): embedding dim
            n_genes (int): number of output genes to regress
            eps (float): error floor for mu, theta, scale params
        """
        super().__init__()
        self.mu = nn.Linear(embed_dim, n_genes)  # mean
        self.theta = nn.Linear(
            embed_dim, n_genes
        )  # inv. dispersion
        self.scale = nn.Linear(
            embed_dim, n_genes
        )  # avg. expr.
        self.gate_logit = nn.Linear(
            embed_dim, n_genes
        )  # zi

        self.eps = eps

    def forward(self, x) -> Dict[str, Float[torch.Tensor, "n_cells n_genes"]]:
        mu = self.mu(x).exp() + self.eps
        theta = self.theta(x).exp() + self.eps
        gate = self.gate_logit(x) 
        scale = self.scale(x).exp() + self.eps
        return dict(mu=mu, theta=theta, zi_logits=gate, scale=scale)
    
class AttnPool(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: Optional[float] = 0.0,
        bias: Optional[bool] = False,
        zero_attn: Optional[bool] = False,
        norm: Optional[nn.Module] = nn.LayerNorm,
    ):
        """A simple attention pool.

        Args:
            embed_dim (int): embedding dim
            num_heads (int): number heads
            dropout (Optional[float], optional): Dropout. Defaults to 0.0.
            bias (Optional[bool], optional): Whether to use bias or not in MLP layers. Defaults to False.
            zero_attn (Optional[bool], optional): Passthrough to MHA. Defaults to False.
            norm (Optional[nn.Module], optional): Passthrough to MHA. Defaults to nn.LayerNorm.
        """
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim,
            bias=bias,
            dropout=dropout,
            num_heads=num_heads,
            add_bias_kv=bias,
            add_zero_attn=zero_attn,
        )
        
        self.norm = norm(embed_dim)
        self.norm_v = norm(embed_dim)
        self.norm_k = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        self.query_token = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.normal_(self.query_token, std=0.02)

        self.dropout_sa = nn.Dropout(dropout)

    def forward(self, x, attn_mask, bs):
        # query_token = query_token.expand(bs, -1)
        query_token = self.norm(self.query_token.expand(bs, -1))

        attn_output, attn_output_weights = self.attn(
            query_token, self.norm_v(x), self.norm_k(x), attn_mask=attn_mask
        )
        attn_output = self.dropout_sa(attn_output)

        return self.norm2(attn_output)
