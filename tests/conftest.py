"""
Shared test fixtures and snapshot infrastructure.

Key design decisions:
  - Synthetic data fixtures are generated programmatically (not committed files)
    so they stay in sync with code changes and are self-documenting.
  - The tensor snapshot fixture writes .pt files to tests/fixtures/snapshots/.
    These get commited to git. 
  - All model tests use a small sized config (dim=32, depth=1) 
  - Spatial data uses a seeded random point cloud (not a grid)
"""

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
import torch

SNAPSHOT_DIR = Path(__file__).parent / "fixtures" / "snapshots"

class TensorSnapshot:
    """Compare tensors against committed .pt reference files.

    On first run (or with --update-snapshots), writes the file and skips.
    On subsequent runs, loads the reference and asserts closeness.
    """

    def __init__(self, test_name: str, update: bool):
        self._test_name = test_name
        self._update = update
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return SNAPSHOT_DIR / f"{self._test_name}__{name}.pt"

    def assert_match(
        self,
        tensor: torch.Tensor,
        name: str,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ):
        path = self._path(name)
        if self._update or not path.exists():
            torch.save(tensor.detach().cpu(), path)
            pytest.skip(f"Snapshot written: {path.name}")
        ref = torch.load(path, weights_only=True)
        torch.testing.assert_close(tensor.detach().cpu(), ref, rtol=rtol, atol=atol)


def pytest_addoption(parser):
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Regenerate all tensor snapshot reference files.",
    )


@pytest.fixture
def snapshot(request):
    update = request.config.getoption("--update-snapshots")
    return TensorSnapshot(request.node.name, update)


@pytest.fixture(scope="session")
def tiny_model_cfg():
    """Dict of kwargs for SpatialCellTransformer.__init__."""
    return dict(
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
        compile=False,
    )


# ---------------------------------------------------------------------------
# Synthetic spatial data fixture — seeded random point cloud
# ---------------------------------------------------------------------------
# Generates a realistic h5ad + parquet pair with random xy coordinates:
#   - 100 cells across 3 tissue sections (unequal sizes)
#   - 10 genes, integer count expression
#   - Seeded RNG for reproducibility
#   - Expression is Poisson-distributed with a different lambda per gene
#     (lambdas log-spaced from 0.5 to 30), mimicking real count data where
#     genes span a wide dynamic range.
#
# All sections share the same spatial range [0, 10] x [0, 10].
# Section isolation is enforced by the loader's section column filtering,
# not by spatial separation.
#
# Tests compare the optimized loader against a brute-force reference
# implementation, so no hardcoded neighbor counts are needed.

N_GENES = 10
N_CELL_TYPES = 5
SECTION_SIZES = {"A": 40, "B": 35, "C": 25} # n cells per section
XY_RANGE = 10.0
# Per-gene Poisson lambdas: log-spaced from 0.5 to 30
GENE_LAMBDAS = np.logspace(np.log10(0.5), np.log10(30), N_GENES)


def brute_force_neighbors(meta_df, center_idx, patch_size, section_col="section"):
    """Reference implementation: find neighbors by brute-force filtering.

    Returns set of integer indices (positional in metadata) that are within
    the patch and in the same section as center_idx.

    Uses the same boundary semantics as CenterMaskNodeOptimized (inclusive <=).
    Exported here so both loader and preprocess tests can use it.
    """
    row = meta_df.iloc[center_idx]
    cx, cy = row["x"], row["y"]
    sec = row[section_col]
    half_x = patch_size[0] / 2.0
    half_y = patch_size[1] / 2.0

    neighbors = set()
    for i in range(len(meta_df)):
        if i == center_idx:
            continue
        r = meta_df.iloc[i]
        if r[section_col] != sec:
            continue
        if abs(r["x"] - cx) <= half_x and abs(r["y"] - cy) <= half_y:
            neighbors.add(i)
    return neighbors


@pytest.fixture
def synthetic_data(tmp_path):
    """Create a paired .h5ad + .parquet with random spatial coordinates.

    Returns (h5ad_path, parquet_path, metadata_df).

    The metadata_df is returned so tests can compute brute-force expected
    neighbors without re-reading the parquet from disk.
    """
    rng = np.random.default_rng(1221)

    all_expr = []
    all_meta = []

    for section_name, n_cells in SECTION_SIZES.items():
        xs = rng.uniform(0, XY_RANGE, size=n_cells)
        ys = rng.uniform(0, XY_RANGE, size=n_cells)
        expr = rng.poisson(lam=GENE_LAMBDAS, size=(n_cells, N_GENES)).astype(np.float32)
        cell_types = rng.integers(0, N_CELL_TYPES, size=n_cells)

        meta = pd.DataFrame({
            "x": xs,
            "y": ys,
            "section": section_name,
            "subclass_int": cell_types,
        })
        all_expr.append(expr)
        all_meta.append(meta)

    expr = np.vstack(all_expr)
    meta = pd.concat(all_meta, ignore_index=True)

    adata = anndata.AnnData(
        X=expr,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(N_GENES)]),
    )

    h5ad_path = tmp_path / "test_data.h5ad"
    pq_path = tmp_path / "test_data.parquet"
    adata.write_h5ad(h5ad_path)
    meta.to_parquet(pq_path, index=False)

    return h5ad_path, pq_path, meta
