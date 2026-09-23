"""
Preprocess functional tests.

Tests the split_sections pipeline against synthetic data to verify:
  1. Round-trip integrity — split then reload; row counts sum correctly
     and positional alignment is preserved.
  2. Transform correctness — log2p1 produces exact expected values.
  3. Dry run — no files written.

Not tested:
  - CLI argument parsing (typer handles that)
  - QC filtering (straightforward boolean mask; low risk)
"""

import anndata
import numpy as np
import pandas as pd
import pytest

from celltransformer.preprocess import (
    Transform,
    split_sections,
    apply_transform,
    find_metadata,
)
from tests.conftest import N_GENES


# ═══════════════════════════════════════════════════════════════════════════
# 1. Round-trip: split → reload → verify
# ═══════════════════════════════════════════════════════════════════════════


class TestSplitRoundTrip:

    def test_split_preserves_total_cells(self, synthetic_data, tmp_path):
        """Sum of cells across per-section files must equal the original."""
        h5ad_path, pq_path, _ = synthetic_data
        adata = anndata.read_h5ad(h5ad_path)
        meta = pd.read_parquet(pq_path)

        out_dir = tmp_path / "split_output"
        split_sections(
            adata=adata,
            metadata=meta,
            section_col="section",
            output_dir=out_dir,
            stem="test",
            transform=None,
            dry_run=False,
        )

        # Reload and count
        total = 0
        for section_h5ad in sorted(out_dir.glob("*.h5ad")):
            section_adata = anndata.read_h5ad(section_h5ad)
            section_pq = section_h5ad.with_suffix(".parquet")
            section_meta = pd.read_parquet(section_pq)

            # Positional alignment: row counts must match
            assert section_adata.n_obs == len(section_meta), (
                f"{section_h5ad.name}: adata has {section_adata.n_obs} rows "
                f"but parquet has {len(section_meta)}"
            )
            total += section_adata.n_obs

        assert total == adata.n_obs

    def test_split_positional_alignment(self, synthetic_data, tmp_path):
        """After splitting, row i in parquet must correspond to row i in X.

        We verify this by checking that the x/y coordinates in the parquet
        match what we'd expect from the section subset of the original data.
        """
        h5ad_path, pq_path, _ = synthetic_data
        adata = anndata.read_h5ad(h5ad_path)
        meta = pd.read_parquet(pq_path)

        out_dir = tmp_path / "split_output"
        split_sections(
            adata=adata,
            metadata=meta,
            section_col="section",
            output_dir=out_dir,
            stem="test",
            transform=None,
            dry_run=False,
        )

        for section_name in meta["section"].unique():
            section_pq = out_dir / f"test_{section_name}.parquet"
            section_h5ad = out_dir / f"test_{section_name}.h5ad"
            section_meta = pd.read_parquet(section_pq)
            section_adata = anndata.read_h5ad(section_h5ad)

            # Original rows for this section
            orig_mask = meta["section"] == section_name
            orig_meta = meta[orig_mask].reset_index(drop=True)

            # x/y coordinates should match exactly
            np.testing.assert_array_equal(
                section_meta["x"].values, orig_meta["x"].values
            )
            np.testing.assert_array_equal(
                section_meta["y"].values, orig_meta["y"].values
            )

            # Expression should match too (no transform applied)
            orig_expr = adata.X[orig_mask.values]
            np.testing.assert_array_almost_equal(
                section_adata.X, orig_expr
            )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Transform correctness
# ═══════════════════════════════════════════════════════════════════════════


class TestTransform:

    def test_log2p1_values(self):
        """apply_transform with log2p1 should produce X = log2(raw + 1)."""
        raw = np.array([[0, 1, 4, 7]], dtype=np.float32)
        adata = anndata.AnnData(X=np.zeros_like(raw))
        adata.layers["raw"] = raw

        apply_transform(adata, Transform.log2p1)

        expected = np.log2(raw + 1.0).astype(np.float32)
        np.testing.assert_array_almost_equal(adata.X, expected)

    def test_log1p_values(self):
        """apply_transform with log1p should produce X = ln(raw + 1)."""
        raw = np.array([[0, 1, 4, 7]], dtype=np.float32)
        adata = anndata.AnnData(X=np.zeros_like(raw))
        adata.layers["raw"] = raw

        apply_transform(adata, Transform.log1p)

        expected = np.log1p(raw).astype(np.float32)
        np.testing.assert_array_almost_equal(adata.X, expected)

    def test_missing_raw_layer_raises(self):
        """apply_transform must raise if .layers['raw'] is absent."""
        adata = anndata.AnnData(X=np.zeros((2, 3), dtype=np.float32))
        with pytest.raises(KeyError, match="raw"):
            apply_transform(adata, Transform.log2p1)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Dry run
# ═══════════════════════════════════════════════════════════════════════════


class TestDryRun:

    def test_dry_run_writes_no_files(self, synthetic_data, tmp_path):
        """With dry_run=True, output_dir should remain empty."""
        h5ad_path, pq_path, _ = synthetic_data
        adata = anndata.read_h5ad(h5ad_path)
        meta = pd.read_parquet(pq_path)

        out_dir = tmp_path / "dry_run_output"
        split_sections(
            adata=adata,
            metadata=meta,
            section_col="section",
            output_dir=out_dir,
            stem="test",
            transform=None,
            dry_run=True,
        )

        # Directory shouldn't even be created in dry-run mode
        assert not out_dir.exists() or len(list(out_dir.iterdir())) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. Metadata discovery
# ═══════════════════════════════════════════════════════════════════════════


class TestFindMetadata:

    def test_finds_parquet(self, synthetic_data):
        """find_metadata should discover the .parquet file next to the .h5ad."""
        h5ad_path, pq_path, _ = synthetic_data
        found = find_metadata(h5ad_path)
        assert found == pq_path

    def test_missing_raises(self, tmp_path):
        """find_metadata should raise FileNotFoundError when no pair exists."""
        fake_h5ad = tmp_path / "orphan.h5ad"
        fake_h5ad.touch()
        with pytest.raises(FileNotFoundError):
            find_metadata(fake_h5ad)
