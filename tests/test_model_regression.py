"""
Model regression snapshots.

Pins the model's numerical output for a fixed seed + config. If a refactor
silently changes the forward pass (e.g., reordering ops, changing init),
these tests fail.

Regenerate snapshots with: pytest --update-snapshots

Separate from test_model.py so you can run non-regression tests quickly
without needing snapshot files or scvi installed.
"""

import pytest
import torch

from celltransformer.model import SpatialCellTransformer


def _make_model_inputs(batch_size: int, n_genes: int, n_cell_types: int):
    lengths = [3 + i for i in range(batch_size)]
    total = sum(lengths)
    return dict(
        expression_mat=torch.randn(total, n_genes),
        cell_type_annot_obs=torch.randint(0, n_cell_types, (total,)),
        lengths=torch.tensor(lengths, dtype=torch.long),
        decoder_cell_type_annot=torch.randint(0, n_cell_types, (batch_size,)),
    )


class TestModelRegression:

    def test_forward_deterministic(self, tiny_model_cfg, snapshot):
        """Full forward pass output must match committed snapshot."""
        torch.manual_seed(0)
        model = SpatialCellTransformer(**tiny_model_cfg)
        model.eval()

        torch.manual_seed(0)
        inputs = _make_model_inputs(3, n_genes=10, n_cell_types=5)

        with torch.no_grad():
            out = model(**inputs)

        snapshot.assert_match(out["nb_params"]["mu"], "forward_mu")

    def test_zinb_loss_deterministic(self, snapshot):
        """zinb_loss_log2 on known inputs must match snapshot.

        This is important because the loss function does a non-obvious
        transform (2^x - 1) before computing NLL. A sign error or base
        change here silently degrades training.
        """
        from celltransformer.distributions import ZeroInflatedNegativeBinomial
        from celltransformer.train import zinb_loss_log2

        torch.manual_seed(0)
        n_batch, n_genes = 4, 10
        zinb = ZeroInflatedNegativeBinomial(
            mu=torch.rand(n_batch, n_genes) * 10 + 0.1,
            theta=torch.rand(n_batch, n_genes) * 5 + 0.1,
            zi_logits=torch.randn(n_batch, n_genes),
        )
        gt_expr = torch.log2(torch.rand(n_batch, n_genes) * 20 + 1)

        loss, _ = zinb_loss_log2(zinb, gt_expr)
        snapshot.assert_match(loss.unsqueeze(0), "zinb_loss")
