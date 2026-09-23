"""ZINB output projection: embedding → (mu, theta, zi_logits, scale) per gene."""

from typing import Dict

import torch
from torch import nn


class ZINBProj(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_genes: int,
        eps: float,
        zero_inflated: bool = True,
        device=None,
        dtype=None,
    ):
        """A utility block to regress the parameters of a (zero-inflated)
        negative binomial distribution from an embedding using a packed projection.

        Args:
            embed_dim (int): embedding dim
            n_genes (int): number of output genes to regress
            eps (float): error floor for mu, theta, scale params
            zero_inflated (bool): If True, uses ZINB (4 params). If False, uses NB (3 params). Default: True
            device: Device to place parameters on. Default: None
            dtype: Data type for parameters. Default: None
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.zero_inflated = zero_inflated
        self.eps = eps
        self.n_genes = n_genes

        # Packed projection: outputs [mu, theta, scale] or [mu, theta, gate_logit, scale]
        n_outputs = 4 if zero_inflated else 3
        self.packed_proj = nn.Linear(
            embed_dim,
            n_genes * n_outputs,
            **factory_kwargs
        )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (..., embed_dim)

        Returns:
            Dictionary with keys: mu, theta, scale, and optionally zi_logits
            Each value has shape (..., n_genes)
        """
        # Apply packed projection and split into components
        result = self.packed_proj(x)  # (..., n_genes * (3 or 4))

        if self.zero_inflated:
            # ZINB: split into 4 components
            mu_logit, theta_logit, gate_logit, scale_logit = torch.chunk(
                result, 4, dim=-1
            )  # Each: (..., n_genes)

            mu = mu_logit.exp() + self.eps
            theta = theta_logit.exp() + self.eps
            gate = gate_logit  # No transformation (logits)
            scale = scale_logit.exp() + self.eps

            return dict(mu=mu, theta=theta, zi_logits=gate, scale=scale)
        else:
            # NB: split into 3 components (no gate)
            mu_logit, theta_logit, scale_logit = torch.chunk(
                result, 3, dim=-1
            )  # Each: (..., n_genes)

            mu = mu_logit.exp() + self.eps
            theta = theta_logit.exp() + self.eps
            scale = scale_logit.exp() + self.eps

            return dict(mu=mu, theta=theta, scale=scale)
