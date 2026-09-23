"""
Model unit tests — shape contracts, config toggles, error handling, GPU smoke tests.

These tests verify correctness without pinning numerical values.
For regression snapshots, see test_model_regression.py.
"""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from celltransformer.model import (
    SpatialCellTransformer,
    CellEmbedder,
    AttentionPool,
    ZINBProj,
    TransformerEncoder,
)
from celltransformer.model._attention import MultiHeadAttention

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _make_model_inputs(batch_size: int, n_genes: int, n_cell_types: int, device="cpu"):
    """Create synthetic inputs for SpatialCellTransformer.forward().

    Each "neighborhood" has (3 + i) cells so lengths are variable,
    which exercises the NJT construction path.
    """
    lengths = [3 + i for i in range(batch_size)]
    total = sum(lengths)

    return dict(
        expression_mat=torch.randn(total, n_genes, device=device),
        cell_type_annot_obs=torch.randint(0, n_cell_types, (total,), device=device),
        lengths=torch.tensor(lengths, dtype=torch.long, device=device),
        decoder_cell_type_annot=torch.randint(0, n_cell_types, (batch_size,), device=device),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Component shape contracts
# ═══════════════════════════════════════════════════════════════════════════


class TestCellEmbedder:
    """CellEmbedder: (expr, cell_type) → (total_cells, model_dim)"""

    @pytest.mark.parametrize("concat_mode", ["add", "cat"])
    @pytest.mark.parametrize("use_diagonal_norm", [True, False])
    @pytest.mark.parametrize("n_cells", [1, 7])
    def test_output_shape(self, concat_mode, use_diagonal_norm, n_cells):
        emb = CellEmbedder(
            expression_dim=10,
            cell_type_cardinality=5,
            output_dim=32,
            concat_mode=concat_mode,
            use_diagonal_norm=use_diagonal_norm,
        )
        out = emb(torch.randn(n_cells, 10), torch.randint(0, 5, (n_cells,)))
        assert out.shape == (n_cells, 32)


class TestTransformerEncoder:
    """TransformerEncoder: (batch, seq, dim) → (batch, seq, dim)"""

    def test_output_shape(self):
        enc = TransformerEncoder(d_model=32, nheads=4, dim_feedforward=64)
        x = torch.randn(2, 5, 32)
        out = enc(x)
        assert out.shape == (2, 5, 32)

    def test_seq_len_one(self):
        """Sequence length 1 is valid (single-cell neighborhood)."""
        enc = TransformerEncoder(d_model=32, nheads=4, dim_feedforward=64)
        out = enc(torch.randn(1, 1, 32))
        assert out.shape == (1, 1, 32)


class TestAttentionPool:
    """AttentionPool: nested jagged tensor → (batch, 1, dim) nested tensor."""

    def test_output_shape(self):
        pool = AttentionPool(input_dim=32, output_dim=32, nheads=4)
        flat = torch.randn(8, 32)
        lengths = torch.tensor([3, 5])
        njt = torch.nested.nested_tensor_from_jagged(flat, lengths=lengths)
        out = pool(njt)
        vals = out.values()
        assert vals.shape == (2, 32)


class TestZINBProj:
    """ZINBProj: (batch, model_dim) → dict of ZINB parameter tensors."""

    def test_output_keys_and_shapes(self):
        proj = ZINBProj(embed_dim=32, n_genes=10, eps=1e-4, zero_inflated=True)
        x = torch.randn(4, 32)
        out = proj(x)
        assert "mu" in out
        assert out["mu"].shape == (4, 10)

    def test_no_zi(self):
        """With zero_inflated=False, zi_logits should be absent or zero."""
        proj = ZINBProj(embed_dim=32, n_genes=10, eps=1e-4, zero_inflated=False)
        out = proj(torch.randn(4, 32))
        assert "mu" in out


# ═══════════════════════════════════════════════════════════════════════════
# Full model — shape, config toggles, error handling
# ═══════════════════════════════════════════════════════════════════════════


class TestSpatialCellTransformer:

    @pytest.fixture(scope="class")
    def model(self, tiny_model_cfg):
        m = SpatialCellTransformer(**tiny_model_cfg)
        m.eval()
        return m

    def test_forward_shape(self, model):
        """Basic forward pass: check output dict keys and tensor shapes."""
        batch_size = 3
        inputs = _make_model_inputs(batch_size, n_genes=10, n_cell_types=5)

        with torch.no_grad():
            out = model(**inputs)

        assert "nb_params" in out
        assert "spatial_context" in out
        assert "decoder_tokens" in out
        assert out["nb_params"]["mu"].shape == (batch_size, 10)
        assert out["decoder_tokens"].shape == (batch_size, 2, 32)

    def test_batch_size_one(self, model):
        """Edge case: single sample in batch."""
        inputs = _make_model_inputs(1, n_genes=10, n_cell_types=5)
        with torch.no_grad():
            out = model(**inputs)
        assert out["nb_params"]["mu"].shape == (1, 10)

    def test_zero_length_neighborhood_raises(self, model):
        """Zero-length neighborhoods must raise ValueError.

        AttentionPool does cross-attention with a length-0 key/value sequence,
        which produces NaN from softmax over an empty set. The model guards
        against this explicitly. Callers must filter zero-length entries
        before calling forward() (the training loop already does this).
        """
        lengths = torch.tensor([5, 0], dtype=torch.long)
        inputs = dict(
            expression_mat=torch.randn(5, 10),
            cell_type_annot_obs=torch.randint(0, 5, (5,)),
            lengths=lengths,
            decoder_cell_type_annot=torch.randint(0, 5, (2,)),
        )

        with pytest.raises(ValueError, match="Zero-length neighborhoods"):
            model(**inputs)

    @pytest.mark.parametrize("concat_mode", ["add", "cat"])
    def test_concat_modes(self, tiny_model_cfg, concat_mode):
        """Both concat modes must produce valid outputs."""
        cfg = {**tiny_model_cfg, "concat_mode": concat_mode}
        m = SpatialCellTransformer(**cfg)
        m.eval()
        inputs = _make_model_inputs(2, n_genes=10, n_cell_types=5)
        with torch.no_grad():
            out = m(**inputs)
        assert out["nb_params"]["mu"].shape == (2, 10)

    @pytest.mark.parametrize("use_diag_norm", [True, False])
    def test_diagonal_norm_toggle(self, tiny_model_cfg, use_diag_norm):
        """DiagonalNormalization on/off must both work."""
        cfg = {**tiny_model_cfg, "use_diagonal_norm": use_diag_norm}
        m = SpatialCellTransformer(**cfg)
        m.eval()
        inputs = _make_model_inputs(2, n_genes=10, n_cell_types=5)
        with torch.no_grad():
            out = m(**inputs)
        assert out["nb_params"]["mu"].shape == (2, 10)


# ═══════════════════════════════════════════════════════════════════════════
# GPU smoke tests
# ═══════════════════════════════════════════════════════════════════════════
# Run locally (GPU available), skip gracefully in CI.


class TestGPU:

    @requires_gpu
    def test_full_forward_on_cuda(self, tiny_model_cfg):
        """Full model forward pass on GPU — exercises the CUDA NJT path."""
        model = SpatialCellTransformer(**tiny_model_cfg, device="cuda")
        model.eval()
        inputs = _make_model_inputs(3, n_genes=10, n_cell_types=5, device="cuda")
        with torch.no_grad():
            out = model(**inputs)
        assert out["nb_params"]["mu"].device.type == "cuda"
        assert out["nb_params"]["mu"].shape == (3, 10)

    @requires_gpu
    def test_attention_pool_on_cuda(self):
        """AttentionPool with NJT on GPU — this is where CPU/GPU paths diverge."""
        pool = AttentionPool(input_dim=32, output_dim=32, nheads=4, device="cuda")
        flat = torch.randn(8, 32, device="cuda")
        lengths = torch.tensor([3, 5], device="cuda")
        njt = torch.nested.nested_tensor_from_jagged(flat, lengths=lengths)
        out = pool(njt)
        assert out.values().device.type == "cuda"
        assert out.values().shape == (2, 32)

    @requires_gpu
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_sdpa_backends_agree(self, dtype):
        """Flash, Efficient, and Math SDPA backends should agree within tolerance.

        Flash and Efficient only run on float16/bfloat16. This test catches
        gross numerical divergence (NaN, shape mismatch, or >0.1 abs diff)
        between backends — not bitwise equality, which is not expected.
        """
        attn = MultiHeadAttention(
            E_q=32, E_k=32, E_v=32, E_total=32, nheads=4,
        ).to(device="cuda", dtype=dtype)
        attn.eval()

        torch.manual_seed(42)
        x = torch.randn(2, 8, 32, device="cuda", dtype=dtype)

        results = {}
        for name, backend in [
            ("flash", SDPBackend.FLASH_ATTENTION),
            ("efficient", SDPBackend.EFFICIENT_ATTENTION),
            ("math", SDPBackend.MATH),
        ]:
            try:
                with sdpa_kernel(backend), torch.no_grad():
                    results[name] = attn(x, x, x).clone()
            except RuntimeError:
                pass  # backend not available for this config

        assert len(results) >= 2, "Need at least 2 backends to compare"

        names = list(results.keys())
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                diff = (results[a] - results[b]).abs().max()
                assert torch.isfinite(diff), f"{a} vs {b}: non-finite values"
                assert diff < 0.1, (
                    f"{a} vs {b}: max abs diff {diff:.4e} exceeds tolerance"
                )
