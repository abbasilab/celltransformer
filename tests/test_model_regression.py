"""
Model regression snapshots.

Pins the model's numerical output for a fixed seed + config. If a refactor
silently changes the forward pass (e.g., reordering ops, changing init),
these tests fail.

Regenerate snapshots with: pytest --update-snapshots
Only regenerate after an intentional change to the model, and say so in the commit.

The forward snapshot was recorded on a GPU (RTX 4090, torch 2.5.1+cu121) and
the test is skipped without CUDA. Weights and inputs are created on CPU and
then moved, so the seeded values do not depend on the GPU's random generator.
The 1e-4 tolerance allows small floating-point differences between GPUs.
It runs at full float32 precision, because importing celltransformer.train
switches matmuls to TF32 (which moves the output by ~1e-3).

Separate from test_model.py so you can run non-regression tests quickly
without needing snapshot files or scvi installed.
"""

import pytest
import torch

from celltransformer.model import SpatialCellTransformer

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


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

    @requires_gpu
    def test_forward_deterministic(self, tiny_model_cfg, snapshot):
        """Full forward pass output must match committed snapshot."""
        #full float32, whatever an earlier import set; restored afterwards
        saved_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            #seed on cpu, then move, so weights and inputs match on any GPU
            torch.manual_seed(0)
            model = SpatialCellTransformer(**tiny_model_cfg).to("cuda")
            model.eval()

            torch.manual_seed(0)
            inputs = _make_model_inputs(3, n_genes=10, n_cell_types=5)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs)
        finally:
            torch.set_float32_matmul_precision(saved_precision)

        snapshot.assert_match(out["nb_params"]["mu"], "forward_mu", rtol=1e-4, atol=1e-4)

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
