"""
Data loader functional tests.

These test the CenterMaskNodeOptimized dataset class against synthetic data
with random spatial coordinates. Because the coordinates are seeded, the
brute-force reference produces deterministic expected results.

Why these tests matter:
  - The loader does spatial filtering, masking, and section isolation.
    Bugs here produce wrong training data with NO error. The model just
    trains on garbage.
  - The binary-search optimization in CenterMaskNodeOptimized is
    correctness-critical — an off-by-one in searchsorted silently drops
    or includes wrong neighbors.
  - Random point clouds exercise floating-point edge cases that regular
    grids miss: points near the exact boundary of patch_size / 2,
    near-identical x-coordinates that stress the binary search.

What we test:
  1. Neighborhood correctness — optimized loader matches brute-force for ALL cells
  2. Masking invariant — center cell in masked_*, never in observed_*
  3. Section isolation — neighbors never cross section boundaries
  4. Edge cases — tiny patch (zero neighbors), dataset length
  5. Collate — batch assembly produces correct shapes and lengths
  6. use_cell_type=False — all labels 0, no cell-type column needed

What we DON'T test:
  - MERFISHDataModule (requires torchdata infra, Lightning, distributed)
  - CellParallelSampler / BudgetBatcher (integration-level; tested via training)
  - AnnDataMetadataNode file discovery (filesystem/glob — fragile to mock)
"""

import numpy as np
import pytest
import torch

from celltransformer.loader.dataset import CenterMaskNodeOptimized
from celltransformer.loader.types import NeighborhoodMetadata
from celltransformer.loader.datamodule import collate
from tests.conftest import (
    brute_force_neighbors,
    N_GENES,
    SECTION_SIZES,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def _make_dataset(h5ad_path, pq_path, patch_size):
    """Construct a CenterMaskNodeOptimized with standard test params."""
    return CenterMaskNodeOptimized(
        metadata_path=str(pq_path),
        adata_path=str(h5ad_path),
        cell_type_colname="subclass_int",
        patch_size=patch_size,
        tissue_section_colname="section",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Neighborhood correctness
# ═══════════════════════════════════════════════════════════════════════════


class TestNeighborhoodCorrectness:
    """Verify that the optimized loader returns the same neighbor count as
    the brute-force reference for every cell in the dataset."""

    def test_all_cells_match_brute_force(self, synthetic_data):
        """For EVERY cell, the optimized loader must return the same number
        of neighbors as the brute-force reference.

        This is the single highest-value test in the suite. It exercises the
        searchsorted boundary logic on random coordinates where points can
        land arbitrarily close to the patch edge.
        """
        h5ad_path, pq_path, meta = synthetic_data
        patch_size = [4.0, 4.0]
        ds = _make_dataset(h5ad_path, pq_path, patch_size)

        mismatches = []
        for idx in range(len(ds)):
            sample = ds[idx]
            expected = brute_force_neighbors(meta, idx, patch_size)
            if sample.num_cells_obs != len(expected):
                mismatches.append(
                    f"Cell {idx}: got {sample.num_cells_obs}, expected {len(expected)}"
                )

        assert not mismatches, (
            f"{len(mismatches)} cells have wrong neighbor counts:\n"
            + "\n".join(mismatches[:10])  # show first 10
        )

    def test_with_different_patch_sizes(self, synthetic_data):
        """Test with a small and a large patch_size to exercise different
        density regimes in the binary search."""
        h5ad_path, pq_path, meta = synthetic_data

        for ps in [[2.0, 2.0], [8.0, 8.0]]:
            ds = _make_dataset(h5ad_path, pq_path, ps)
            for idx in range(len(ds)):
                sample = ds[idx]
                expected = brute_force_neighbors(meta, idx, ps)
                assert sample.num_cells_obs == len(expected), (
                    f"patch_size={ps}, cell {idx}: "
                    f"got {sample.num_cells_obs}, expected {len(expected)}"
                )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Masking invariant
# ═══════════════════════════════════════════════════════════════════════════


class TestMaskingInvariant:
    """The center cell's expression must be in masked_expression and must
    NOT appear in observed_expression."""

    def test_masked_expression_matches_center(self, synthetic_data):
        """masked_expression should equal the center cell's row in X."""
        h5ad_path, pq_path, meta = synthetic_data
        ds = _make_dataset(h5ad_path, pq_path, [4.0, 4.0])

        # Check a sample of cells across all sections
        for idx in [0, 10, 39, 40, 60, 75, 99]:
            sample = ds[idx]
            expected_expr = torch.from_numpy(np.atleast_2d(ds.adata.X[idx]))
            torch.testing.assert_close(sample.masked_expression, expected_expr)

    def test_center_not_in_observed(self, synthetic_data):
        """The center cell's expression vector must not appear in observed_expression.

        With random integer expression values, exact row equality is a
        reliable check (probability of two random cells having identical
        expression across 10 genes with values in [0,50) is negligible).
        """
        h5ad_path, pq_path, meta = synthetic_data
        ds = _make_dataset(h5ad_path, pq_path, [4.0, 4.0])

        for idx in range(len(ds)):
            sample = ds[idx]
            if sample.num_cells_obs == 0:
                continue
            masked = sample.masked_expression    # (1, n_genes)
            observed = sample.observed_expression  # (n_neighbors, n_genes)
            matches = (observed == masked).all(dim=1)
            assert not matches.any(), (
                f"Cell {idx}: center expression found in observed_expression"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Section isolation
# ═══════════════════════════════════════════════════════════════════════════


class TestSectionIsolation:
    """Neighbors must never cross section boundaries, even with a huge patch_size."""

    def test_large_patch_no_cross_section(self, synthetic_data):
        """With patch_size large enough to cover all cells, neighbors
        should still only come from the center cell's own section.

        All sections share the same [0, 10] spatial range, so without
        section filtering every cell would see cells from all sections.
        """
        h5ad_path, pq_path, meta = synthetic_data
        ds = _make_dataset(h5ad_path, pq_path, [500.0, 500.0])

        # For each section, a cell should see exactly (section_size - 1) neighbors
        section_start = 0
        for section_name, section_size in SECTION_SIZES.items():
            idx = section_start  # first cell in this section
            sample = ds[idx]
            assert sample.num_cells_obs == section_size - 1, (
                f"Section {section_name}: cell {idx} got {sample.num_cells_obs} "
                f"neighbors, expected {section_size - 1}"
            )
            section_start += section_size


# ═══════════════════════════════════════════════════════════════════════════
# 4. Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:

    def test_tiny_patch_no_neighbors(self, synthetic_data):
        """A very small patch_size should yield zero (or very few) neighbors.
        The dataset must return valid NeighborhoodMetadata without crashing."""
        h5ad_path, pq_path, meta = synthetic_data
        # 0.001 is much smaller than typical inter-cell distance in [0, 10]
        ds = _make_dataset(h5ad_path, pq_path, [0.001, 0.001])

        sample = ds[0]
        # We can't assert exactly 0 (two random points could be very close),
        # but we can verify the structure is valid
        assert isinstance(sample, NeighborhoodMetadata)
        assert sample.masked_expression.shape == (1, N_GENES)

    def test_dataset_length(self, synthetic_data):
        """__len__ should equal total cells across all sections."""
        h5ad_path, pq_path, meta = synthetic_data
        ds = _make_dataset(h5ad_path, pq_path, [4.0, 4.0])
        assert len(ds) == sum(SECTION_SIZES.values())


# ═══════════════════════════════════════════════════════════════════════════
# 5. Collate function
# ═══════════════════════════════════════════════════════════════════════════


class TestCollate:
    """The collate function assembles a list of NeighborhoodMetadata into
    the dict that the model's training_step expects."""

    def test_collate_shapes(self, synthetic_data):
        """Collated batch should have correct concatenated shapes."""
        h5ad_path, pq_path, meta = synthetic_data
        ds = _make_dataset(h5ad_path, pq_path, [4.0, 4.0])

        # Grab a few samples that are likely to have neighbors
        samples = [ds[i] for i in [0, 10, 50]]
        # Filter to samples with neighbors (collate skips empty ones)
        samples = [s for s in samples if s.num_cells_obs > 0]
        assert len(samples) > 0, "Need at least one sample with neighbors"

        batch = collate(samples)

        n_samples = len(samples)
        total_neighbors = sum(s.num_cells_obs for s in samples)

        # lengths: one per sample
        assert batch["lengths"].shape == (n_samples,)
        assert int(batch["lengths"].sum()) == total_neighbors

        # expression_mat: all neighbors concatenated
        assert batch["expression_mat"].shape == (total_neighbors, N_GENES)

        # decoder_cell_type_annot: one per sample (the masked cell's type)
        assert batch["decoder_cell_type_annot"].shape == (n_samples,)

        # gt_expr: one per sample (the masked cell's expression)
        assert batch["gt_expr"].shape == (n_samples, N_GENES)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Cell types switched off (use_cell_type=False)
# ═══════════════════════════════════════════════════════════════════════════


class TestUseCellTypeOff:
    """With use_cell_type=False every cell gets label 0 and the cell-type
    column is never read, so data without cell types can be used."""

    def test_all_labels_zero(self, synthetic_data):
        """Every observed and masked label is 0; neighborhoods are unchanged."""
        h5ad_path, pq_path, meta = synthetic_data
        ds_on = _make_dataset(h5ad_path, pq_path, [4.0, 4.0])
        ds_off = CenterMaskNodeOptimized(
            metadata_path=str(pq_path),
            adata_path=str(h5ad_path),
            cell_type_colname="subclass_int",
            patch_size=[4.0, 4.0],
            tissue_section_colname="section",
            use_cell_type=False,
        )

        for i in range(len(ds_off)):
            on, off = ds_on[i], ds_off[i]
            assert off.masked_cell_type.tolist() == [0]
            assert (off.observed_cell_type == 0).all()
            assert off.num_cells_obs == on.num_cells_obs

    def test_no_cell_type_column(self, synthetic_data, tmp_path):
        """A parquet with no cell-type column loads when cell types are off."""
        h5ad_path, pq_path, meta = synthetic_data
        pq_no_ct = tmp_path / "no_cell_type.parquet"
        meta.drop(columns=["subclass_int"]).to_parquet(pq_no_ct, index=False)

        ds = CenterMaskNodeOptimized(
            metadata_path=str(pq_no_ct),
            adata_path=str(h5ad_path),
            cell_type_colname=None,
            patch_size=[4.0, 4.0],
            tissue_section_colname="section",
            use_cell_type=False,
        )
        sample = ds[0]
        assert sample.masked_cell_type.tolist() == [0]
