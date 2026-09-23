"""Entry point: celltransformer-profile / python -m celltransformer.profile

Profile the cell-type composition of K-means clusters or CCF brain regions
for a single mouse / embeddings directory.

Two modes:
  - **K-means mode** (label_file provided): Load a cluster label .npy file and profile
    cell-type composition per cluster.
  - **CCF mode** (no label_file): Use a CCF brain-region column from parquet metadata
    to define regions.

Pipeline:
  1. Glob *_embeddings.npz in embeddings_dir for stem discovery only (no np.load).
     Read the co-located .parquet files for metadata.
  2. Load taxonomy YAML -> ordered cell-type list for deterministic column ordering.
  3. Map each cell's cell_type_col value -> integer index via taxonomy order.
  4. Build contingency matrix (regions x cell types) and save as .npy.

Outputs:
  - K-means mode: {tag}_profile_k{K}.npy, {tag}_celltypes.txt
  - CCF mode: {tag}_profile_ccf_{ccf_col}.npy, {tag}_ccf_regions.txt, {tag}_celltypes.txt

Example:
    # K-means mode — single mouse
    celltransformer-profile \\
        /data2/alexlee/embeddings/20260305_094827/mouse_687997_ps160 \\
        --label-file /path/to/mouse_687997_kmeans_k100_labels.npy \\
        --output-dir /tmp/profile --tag mouse687997

    # CCF mode (no --label-file)
    celltransformer-profile \\
        /data2/alexlee/embeddings/20260305_094827/mouse_687997_ps160 \\
        --output-dir /tmp/profile --tag mouse687997_ccf
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
import typer
from numpy.typing import NDArray


def load_taxonomy(yaml_path: str) -> list[str]:
    """Load ordered cell-type list from a taxonomy YAML file."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def build_contingency(
    region_ids: NDArray[np.intp],
    celltype_ids: NDArray[np.intp],
    n_regions: int,
    n_types: int,
) -> NDArray[np.int64]:
    """Build (n_regions, n_types) contingency matrix from parallel id arrays."""
    out = np.zeros((n_regions, n_types), dtype=np.int64)
    np.add.at(out, (region_ids, celltype_ids), 1)
    return out


def main(
    embeddings_dir: str,
    label_file: Optional[str] = None,
    output_dir: str = ".",
    tag: str = "profile",
    cell_type_col: str = "RF_2lvl_consensus_subclass",
    ccf_col: str = "CCF_level2",
    taxonomy_yaml: str = "/data1/alexlee/spatial/data/spatial/taxonomy/scvi_375_subclass_taxonomy.yaml",
    section_col: str = "brain_section_label",
) -> None:
    """Profile cell-type composition of K-means clusters or CCF brain regions.

    Args:
        embeddings_dir: Directory containing *_embeddings.npz + co-located .parquet files.
        label_file: .npy cluster label file (K-means mode). Omit for CCF mode.
        output_dir: Where to write output .npy and .txt files.
        tag: Prefix for output filenames.
        cell_type_col: Parquet column for cell-type labels.
        ccf_col: Parquet column for CCF brain region (used in CCF mode).
        taxonomy_yaml: Path to YAML listing cell types in canonical order.
        section_col: Parquet column for tissue section identifier.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    is_kmeans = label_file is not None

    # ------------------------------------------------------------------ #
    # 1. Discover sections via npz stems, load parquet metadata only
    # ------------------------------------------------------------------ #
    emb_dir = Path(embeddings_dir)
    cols_to_load = [cell_type_col, section_col]
    if not is_kmeans:
        cols_to_load.append(ccf_col)

    npz_paths = sorted(emb_dir.glob("*_embeddings.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No *_embeddings.npz files found in {emb_dir}")

    meta_rows = []
    for npz_path in npz_paths:
        stem = npz_path.stem.removesuffix("_embeddings")
        parquet_path = npz_path.with_name(f"{stem}.parquet")
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing parquet: {parquet_path}")
        meta_rows.append(pd.read_parquet(parquet_path, columns=cols_to_load))

    meta_df = pd.concat(meta_rows, axis=0, ignore_index=True)
    total_cells = len(meta_df)
    print(f"[{emb_dir.name}] {len(npz_paths)} sections, {total_cells:,} cells")

    # ------------------------------------------------------------------ #
    # 2. Load taxonomy and build cell-type index
    # ------------------------------------------------------------------ #
    taxonomy = load_taxonomy(taxonomy_yaml)
    celltype_to_idx = {name: i for i, name in enumerate(taxonomy)}

    data_types = set(meta_df[cell_type_col].unique())
    missing = data_types - set(celltype_to_idx)
    if missing:
        missing_sorted = sorted(missing)
        print(f"WARNING: {len(missing_sorted)} cell types in data not in taxonomy (appending): {missing_sorted[:10]}...")
        for name in missing_sorted:
            celltype_to_idx[name] = len(taxonomy)
            taxonomy.append(name)

    n_types = len(taxonomy)

    # ------------------------------------------------------------------ #
    # 3. Map cell types to integer indices
    # ------------------------------------------------------------------ #
    celltype_ids = meta_df[cell_type_col].map(celltype_to_idx).values.astype(np.intp)
    section_labels = meta_df[section_col].values

    celltypes_path = out / f"{tag}_celltypes.txt"
    with open(celltypes_path, "w") as f:
        for name in taxonomy:
            f.write(name + "\n")
    print(f"Saved {celltypes_path} ({n_types} types)")

    # ------------------------------------------------------------------ #
    # 4. Build contingency matrix
    # ------------------------------------------------------------------ #
    if is_kmeans:
        labels = np.load(label_file)
        if labels.shape[0] != total_cells:
            raise ValueError(
                f"{label_file}: label count ({labels.shape[0]}) != total cells ({total_cells})"
            )
        k = int(labels.max()) + 1
        region_ids = labels.astype(np.intp)

        contingency = build_contingency(region_ids, celltype_ids, k, n_types)
        npy_path = out / f"{tag}_profile_k{k}.npy"
        np.save(npy_path, contingency)
        print(f"Saved {npy_path}  shape={contingency.shape}")
    else:
        # CCF mode
        regions_raw = meta_df[ccf_col].astype(str).fillna("unknown").values
        unique_regions = sorted(set(regions_raw))
        region_to_idx = {r: i for i, r in enumerate(unique_regions)}
        n_regions = len(unique_regions)
        region_ids = np.array([region_to_idx[r] for r in regions_raw], dtype=np.intp)

        contingency = build_contingency(region_ids, celltype_ids, n_regions, n_types)
        npy_path = out / f"{tag}_profile_ccf_{ccf_col}.npy"
        np.save(npy_path, contingency)
        print(f"Saved {npy_path}  shape={contingency.shape}")

        regions_path = out / f"{tag}_ccf_regions.txt"
        with open(regions_path, "w") as f:
            for r in unique_regions:
                f.write(str(r) + "\n")
        print(f"Saved {regions_path} ({n_regions} regions)")


def cli():
    typer.run(main)


if __name__ == "__main__":
    cli()
