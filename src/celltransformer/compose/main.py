"""Entry point: celltransformer-compose / python -m celltransformer.compose

Build a cluster-to-cell-type composition table from clustering labels and
spatial metadata parquets.

Given a .npy label file (one integer cluster ID per cell) and a directory of
.parquet metadata files, builds an (n_clusters x n_cell_types) contingency
matrix of raw integer counts and writes it to CSV.

Optionally filters cells with zero spatial neighbours using `has_neighbors`
boolean arrays from --has-neighbors-path .npy file.

Cell type encoding uses sklearn LabelEncoder (which always sorts classes
alphabetically). When a taxonomy YAML is provided (flat list of label strings),
LabelEncoder is fit on that list — ensuring a consistent column *set* across
runs even when some types are absent from the data.

Label path must be .npy format.

Output CSV columns: cluster, n_cells, <cell_type_1>, <cell_type_2>, ...
Each row is one cluster. Values are raw counts (not normalised).

Pipeline:
  1. np.load(label_path) → integer cluster labels
  2. (Optional) Load has_neighbors mask from --has-neighbors-path .npy file
  3. Glob parquet_dir/*.parquet, read [cell_type_col], concatenate
  4. LabelEncoder or taxonomy YAML → integer-encode cell types
  5. Error on NaN or unknown cell types
  6. np.add.at → (n_clusters, n_cell_types) contingency matrix (uint32)
  7. Save CSV + sidecar .meta.yaml

Example:
    celltransformer-compose \\
        /data2/alexlee/embeddings/20260227_203226/mouse_638850 \\
        /data2/alexlee/embeddings/20260227_203226/clustering_638850/638850_kmeans_k1000_labels.npy \\
        --output-dir ./outputs

    # With taxonomy and neighbor filtering:
    celltransformer-compose \\
        /data2/alexlee/embeddings/20260227_203226/mouse_638850 \\
        /path/to/labels.npy \\
        --cell-type-yaml /data1/alexlee/spatial/data/spatial/taxonomy/scvi_375_subclass_taxonomy.yaml \\
        --has-neighbors-path /path/to/638850_has_neighbors.npy

Exports:
  main  — typed entry point, callable without CLI context
  cli   — typer wrapper
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
import yaml
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)


def main(
    parquet_dir: str,
    label_path: str,
    output_dir: str = ".",
    cell_type_col: str = typer.Option("RF_2lvl_consensus_subclass"),
    cell_type_yaml: Optional[str] = typer.Option(
        None,
        help="Taxonomy YAML (flat list of cell type labels). "
        "Fixes column set (ordering is alphabetical via LabelEncoder); without this, uses LabelEncoder on data.",
    ),
    has_neighbors_path: Optional[str] = typer.Option(
        None,
        help="Path to has_neighbors .npy boolean mask (saved by celltransformer-cluster). "
        "Filters out cells with zero spatial neighbors.",
    ),
    overwrite: bool = typer.Option(False, help="Overwrite output CSV if it exists."),
    dry_run: bool = typer.Option(False),
) -> None:
    """Build cluster-to-cell-type composition CSV from labels + parquet metadata.

    Args:
        parquet_dir:    Directory with *.parquet files containing cell_type_col.
        label_path:     .npy file with integer cluster labels (one per cell).
        output_dir:     Where to write the output CSV.
        cell_type_col:  Parquet column with cell type annotations.
        cell_type_yaml: Taxonomy YAML file (flat list). Fixes column set (order is alphabetical).
        has_neighbors_path: .npy boolean mask from celltransformer-cluster for neighbor filtering.
        overwrite:      If True, overwrite existing output. If False, error if exists.
        dry_run:        Print summary and exit without writing.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Check output path early
    out = Path(output_dir)
    out_path = out / "composition.csv"
    meta_path = out / "composition.meta.yaml"
    if (out_path.exists() or meta_path.exists()) and not overwrite and not dry_run:
        raise FileExistsError(
            f"{out_path} or {meta_path} already exists. Use --overwrite to replace."
        )

    # 1. Load cluster labels
    labels = np.load(label_path)
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    log.info(f"Loaded {len(labels):,} cells, {n_clusters} clusters (IDs {unique_clusters.min()}–{unique_clusters.max()}) from {label_path}")

    # 2. Load has_neighbors mask (optional)
    has_neighbors_mask: Optional[np.ndarray] = None
    if has_neighbors_path is not None:
        has_neighbors_mask = np.load(has_neighbors_path)
        n_no_neighbors = int((~has_neighbors_mask).sum())
        log.info(f"has_neighbors: {has_neighbors_mask.sum():,}/{len(has_neighbors_mask):,} cells have neighbors ({n_no_neighbors} without)")
        if len(has_neighbors_mask) != len(labels):
            raise ValueError(
                f"has_neighbors length ({len(has_neighbors_mask):,}) "
                f"!= label count ({len(labels):,})"
            )

    # 3. Read cell type column from all parquets
    dfs = []
    for pq in sorted(Path(parquet_dir).glob("*.parquet")):
        dfs.append(pd.read_parquet(pq, columns=[cell_type_col]))
    if not dfs:
        raise FileNotFoundError(f"No .parquet files in {parquet_dir}")
    cell_types = pd.concat(dfs, ignore_index=True)[cell_type_col]

    if len(cell_types) != len(labels):
        raise ValueError(
            f"Parquet rows ({len(cell_types):,}) != label count ({len(labels):,})"
        )

    # Apply has_neighbors filter
    if has_neighbors_mask is not None:
        try:
            cell_types = cell_types[has_neighbors_mask].reset_index(drop=True)
            labels = labels[has_neighbors_mask]
        except (IndexError, KeyError) as e:
            raise ValueError(
                f"Boolean mask from {has_neighbors_path} (length {len(has_neighbors_mask):,}) "
                f"failed to index cell_types (length {len(cell_types):,}) or "
                f"labels (length {len(labels):,}). Ensure the mask, parquets, and "
                f"label .npy all come from the same extraction run."
            ) from e
        log.info(f"After neighbor filter: {len(labels):,} cells")

    # 4. Error on NaN cell types
    nan_mask = cell_types.isna()
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        raise ValueError(
            f"{n_nan:,} cells have NaN in '{cell_type_col}'. "
            f"Inspect your data — all cells must have valid cell type annotations."
        )

    # 5. Integer-encode cell types via LabelEncoder
    le = LabelEncoder()
    if cell_type_yaml is not None:
        with open(cell_type_yaml) as f:
            taxonomy = yaml.safe_load(f)
        le.fit(taxonomy)
        log.info(f"Taxonomy: {len(taxonomy)} cell types from {Path(cell_type_yaml).name}")

        # Error on cell types not in taxonomy
        unknown = sorted(set(cell_types.unique()) - set(taxonomy))
        if unknown:
            raise ValueError(
                f"{len(unknown)} cell type(s) in data not found in taxonomy "
                f"{Path(cell_type_yaml).name}:\n  {unknown}\n"
                f"Update the taxonomy or check the cell_type_col."
            )
    else:
        le.fit(cell_types.values)
        log.info(f"{len(le.classes_)} unique cell types discovered via LabelEncoder from '{cell_type_col}'")

    ct_ints = le.transform(cell_types.values)
    all_sub = le.classes_.tolist()
    n_sub = len(all_sub)

    # 6. Build (n_clusters, n_sub) contingency matrix
    # Map cluster labels to dense row indices
    row_indices = np.searchsorted(unique_clusters, labels)
    comp = np.zeros((n_clusters, n_sub), dtype=np.uint32)
    np.add.at(comp, (row_indices, ct_ints), 1)
    row_counts = comp.sum(axis=1)

    if row_counts.max() == 0:
        raise ValueError(
            "All clusters have zero cells after filtering. "
            "Check that the labels, parquets, and taxonomy are consistent."
        )

    log.info(
        f"Contingency: {n_clusters} clusters × {n_sub} cell types, cluster sizes: "
        f"min={row_counts.min()}, median={int(np.median(row_counts))}, max={row_counts.max()}"
    )

    if dry_run:
        log.info(f"[dry-run] Would write {n_clusters} rows × {n_sub + 2} columns to {out_path}")
        return

    # 7. Write CSV
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(comp, columns=all_sub)
    df["cluster"] = unique_clusters
    df["n_cells"] = row_counts
    # Reorder so cluster and n_cells come first
    cols = ["cluster", "n_cells"] + all_sub
    df = df[cols]

    # Write CSV (clean, no comment header)
    df.to_csv(out_path, index=False)

    # Write sidecar metadata YAML
    taxonomy_name = Path(cell_type_yaml).name if cell_type_yaml else "LabelEncoder"
    neighbor_filtered = has_neighbors_path is not None
    meta = {
        "parquet_dir": parquet_dir,
        "label_path": label_path,
        "cell_type_col": cell_type_col,
        "taxonomy": taxonomy_name,
        "neighbor_filtered": neighbor_filtered,
        "n_clusters": n_clusters,
        "n_cell_types": n_sub,
        "total_cells": int(row_counts.sum()),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    log.info(f"Saved {out_path} + {meta_path}  ({n_clusters} clusters × {n_sub} cell types)")


def cli():
    typer.run(main)


if __name__ == "__main__":
    cli()
