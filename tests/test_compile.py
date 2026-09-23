"""Correctness tests for torch.compile: compiled model must match eager output.

Each test creates two models with identical weights — one eager, one compiled —
and asserts their outputs are close. This catches silent numerical divergence
from operator fusion or graph-break miscompilation.

All tests require CUDA (compile + nested jagged tensors).
"""

import pytest
import torch

from celltransformer.model import SpatialCellTransformer

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for compile tests"
)

# Small model config for fast compile + forward
TINY_CFG = dict(
    expression_dim=10,
    cell_type_cardinality=5,
    model_dim=32,
    encoder_depth=1,
    decoder_depth=1,
    num_heads=4,
    ffn_dim=64,
    concat_mode="add",
    use_zi=True,
    use_diagonal_norm=True,
    dropout=0.0,
)


def _make_inputs(batch_size: int, n_genes: int, n_cell_types: int, device="cuda"):
    """Deterministic model inputs on device."""
    torch.manual_seed(42)
    # Variable-length neighborhoods: 3, 5, 4 neighbors
    neighbor_counts = [3, 5, 4][:batch_size]
    total_neighbors = sum(neighbor_counts)

    expression_mat = torch.randn(total_neighbors, n_genes, device=device)
    cell_type_annot_obs = torch.randint(0, n_cell_types, (total_neighbors,), device=device)
    lengths = torch.tensor(neighbor_counts, dtype=torch.long, device=device)
    decoder_cell_type_annot = torch.randint(0, n_cell_types, (batch_size,), device=device)

    return dict(
        expression_mat=expression_mat,
        cell_type_annot_obs=cell_type_annot_obs,
        lengths=lengths,
        decoder_cell_type_annot=decoder_cell_type_annot,
    )


def _build_and_copy(cfg: dict, compile: bool, device="cuda"):
    """Build model with compile flag, move to device."""
    model = SpatialCellTransformer(**cfg, compile=compile, device=device, dtype=torch.float32)
    model.eval()
    return model


@requires_cuda
class TestCompileCorrectness:
    """Compiled model output must match eager output within tolerance."""

    def test_forward_matches_eager_f32(self):
        """f32: compiled forward matches eager forward."""
        device = "cuda"
        torch.manual_seed(0)
        eager = _build_and_copy(TINY_CFG, compile=False, device=device)

        torch.manual_seed(0)
        compiled = _build_and_copy(TINY_CFG, compile=True, device=device)

        inputs = _make_inputs(3, n_genes=10, n_cell_types=5, device=device)

        with torch.no_grad():
            ref = eager(**inputs)
            out = compiled(**inputs)

        for key in ["spatial_context", "decoder_tokens"]:
            torch.testing.assert_close(
                out[key].values() if out[key].is_nested else out[key],
                ref[key].values() if ref[key].is_nested else ref[key],
                rtol=1e-4, atol=1e-4,
                msg=f"compile mismatch on {key} (f32)",
            )

        for param_key in ref["nb_params"]:
            torch.testing.assert_close(
                out["nb_params"][param_key],
                ref["nb_params"][param_key],
                rtol=1e-3, atol=1e-3,
                msg=f"compile mismatch on nb_params.{param_key} (f32)",
            )

    def test_forward_matches_eager_bf16(self):
        """bf16: compiled forward matches eager forward."""
        device = "cuda"
        torch.manual_seed(0)
        eager = _build_and_copy(TINY_CFG, compile=False, device=device)

        torch.manual_seed(0)
        compiled = _build_and_copy(TINY_CFG, compile=True, device=device)

        inputs = _make_inputs(3, n_genes=10, n_cell_types=5, device=device)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            ref = eager(**inputs)
            out = compiled(**inputs)

        for key in ["spatial_context", "decoder_tokens"]:
            torch.testing.assert_close(
                out[key].values() if out[key].is_nested else out[key],
                ref[key].values() if ref[key].is_nested else ref[key],
                rtol=1e-2, atol=1e-2,
                msg=f"compile mismatch on {key} (bf16)",
            )

        for param_key in ref["nb_params"]:
            torch.testing.assert_close(
                out["nb_params"][param_key],
                ref["nb_params"][param_key],
                rtol=1e-2, atol=1e-2,
                msg=f"compile mismatch on nb_params.{param_key} (bf16)",
            )

    @pytest.mark.xfail(
        reason="torch.compile on whole model fails with NJT ops in PyTorch 2.5 (inductor backend). "
               "Revisit when PyTorch NJT compile support improves.",
        raises=Exception,
    )
    def test_whole_model_compile_matches_eager_f32(self):
        """torch.compile(model) on the whole model matches eager."""
        device = "cuda"
        torch.manual_seed(0)
        eager = _build_and_copy(TINY_CFG, compile=False, device=device)

        torch.manual_seed(0)
        whole_compiled = _build_and_copy(TINY_CFG, compile=False, device=device)
        whole_compiled = torch.compile(whole_compiled)

        inputs = _make_inputs(3, n_genes=10, n_cell_types=5, device=device)

        with torch.no_grad():
            ref = eager(**inputs)
            out = whole_compiled(**inputs)

        for key in ["spatial_context", "decoder_tokens"]:
            torch.testing.assert_close(
                out[key].values() if out[key].is_nested else out[key],
                ref[key].values() if ref[key].is_nested else ref[key],
                rtol=1e-4, atol=1e-4,
            )

    def test_backward_runs_without_error(self):
        """Compiled model can compute gradients without crashing."""
        device = "cuda"
        torch.manual_seed(0)
        model = _build_and_copy(TINY_CFG, compile=True, device=device)
        model.train()

        inputs = _make_inputs(3, n_genes=10, n_cell_types=5, device=device)
        out = model(**inputs)

        # Use mu as a proxy loss
        loss = out["nb_params"]["mu"].mean()
        loss.backward()

        # Verify at least some gradients are non-zero
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No non-zero gradients found after backward"
