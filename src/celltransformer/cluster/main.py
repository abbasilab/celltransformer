"""Entry point: celltransformer-cluster / python -m celltransformer.cluster

Cluster 384-dim spatial-context embeddings for one or more mice using cuML KMeans,
then visualize cluster labels as spatial scatter plots per brain section.

Pipeline:
  1. For each embeddings_dir, load all *_embeddings.npz files (one per section).
  2. Load corresponding .parquet metadata files (must be co-located with the .npz files)
     to get center_x/center_y/section.
  3. Concatenate all specimens into a single (N, 384) float32 array + metadata DataFrame.
  4. For each k in ks: run cuML KMeans(k) on GPU, save labels, plot all sections.
     Embeddings are loaded once and reused across all k values.
  5. Labels saved to <output_base>/<tag>_kmeans_k{k}_labels.npy per k.
  6. Plots saved to <output_base>/<tag>_cluster_plots_k{k}/ per k.

Example:
    # Single specimen, single GPU
    celltransformer-cluster --ks 100 --ks 1000

    # Multiple specimens, 4 GPUs (recommended for >1M cells)
    celltransformer-cluster \\
        --embeddings-dirs /data2/alexlee/embeddings/638850 \\
        --embeddings-dirs /data2/alexlee/embeddings/687997 \\
        --tag 638850_687997 \\
        --ks 100 --ks 1000 \\
        --n-gpus 4
"""

import random
from pathlib import Path
from typing import List, Optional

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

from celltransformer.plotting import make_section_plot


# ---------------------------------------------------------------------------
# KMeans fitting
# ---------------------------------------------------------------------------

def _fit_single_gpu(embeddings: np.ndarray, k: int, random_state: int) -> np.ndarray:
    from cuml.cluster import KMeans  # type: ignore

    print(f"Running cuML KMeans (k={k}, n_gpus=1, random_state={random_state}) ...")
    km = KMeans(n_clusters=k, random_state=random_state, n_init=5)
    labels = km.fit_predict(embeddings)
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)
    return labels.astype(np.int32)


def _fit_multi_gpu(
    embeddings: np.ndarray, k: int, n_gpus: int, random_state: int
) -> np.ndarray:
    import cupy as cp
    import cudf
    import dask_cudf
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client, wait
    from cuml.dask.cluster import KMeans as DaskKMeans  # type: ignore

    print(f"Running cuML Dask KMeans (k={k}, n_gpus={n_gpus}, random_state={random_state}) ...")
    cluster = LocalCUDACluster(n_workers=n_gpus, threads_per_worker=1)
    client = Client(cluster)
    try:
        X_dask = dask_cudf.from_cudf(cudf.DataFrame(embeddings), npartitions=n_gpus)
        X_dask = client.persist(X_dask)
        wait(X_dask)

        km = DaskKMeans(n_clusters=k, random_state=random_state, client=client)
        km.fit(X_dask)
        return cp.asnumpy(km.labels_).astype(np.int32)
    finally:
        client.close()
        cluster.close()


# ---------------------------------------------------------------------------
# Cluster + plot pipeline
# ---------------------------------------------------------------------------

def _cluster_and_plot(
    embeddings: np.ndarray,
    meta_df: pd.DataFrame,
    k: int,
    base_out: Path,
    tag: str,
    random_state: int,
    s: float,
    x_col: str = "center_x",
    y_col: str = "center_y",
    section_col: str = "section",
    n_gpus: int = 1,
    no_plots: bool = False,
) -> None:
    """Run KMeans for one k, save labels, and optionally render section plots.

    When n_gpus == 1, uses cuml.cluster.KMeans on a single GPU.
    When n_gpus > 1, spins up a LocalCUDACluster with n_gpus workers and uses
    cuml.dask.cluster.KMeans. Recommended for datasets above ~1M cells.
    """
    print(f"\n{'='*60}")

    if n_gpus > 1:
        labels = _fit_multi_gpu(embeddings, k, n_gpus, random_state)
    else:
        labels = _fit_single_gpu(embeddings, k, random_state)

    print(f"Clustering done. Labels shape: {labels.shape}, range: [{labels.min()}, {labels.max()}]")

    labels_path = base_out / f"{tag}_kmeans_k{k}_labels.npy"
    np.save(labels_path, labels)
    print(f"Saved labels → {labels_path}")

    if no_plots:
        return

    # Colour mapping
    _cm = sns.color_palette(cc.glasbey, n_colors=k)
    for idx in range(5):
        random.Random(random_state + idx).shuffle(_cm)
    hex_values = {i: v for i, v in zip(range(k), _cm.as_hex())}

    plot_df = meta_df.copy()
    plot_df["cluster_label"] = labels
    plot_df["datadrivencolor"] = plot_df["cluster_label"].map(hex_values)

    output_dir = base_out / f"{tag}_cluster_plots_k{k}"
    output_dir.mkdir(parents=True, exist_ok=True)
    sections = plot_df[section_col].unique()
    print(f"Plotting {len(sections)} sections → {output_dir}")

    for section in sections:
        save_path = output_dir / f"{section}_k{k}.png"
        fig, _ = make_section_plot(
            section_name=section,
            df=plot_df,
            x_col=x_col,
            y_col=y_col,
            color_col="datadrivencolor",
            section_col=section_col,
            s=s,
            title=True,
            save=str(save_path),
        )
        plt.close(fig)
        print(f"  Saved {save_path.name}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(
    embeddings_dirs: Optional[List[str]] = None,
    output_base: str = "/data2/alexlee/embeddings",
    tag: str = "638850",
    ks: Optional[List[int]] = None,
    n_gpus: int = typer.Option(1, help="Number of GPUs for KMeans. Use >1 for multi-GPU (recommended above ~1M cells)."),
    random_state: int = 42,
    s: float = 1.5,
    x_col: str = "x",
    y_col: str = "y",
    section_col: str = "brain_section_label",
    dry_run: bool = False,
    no_plots: bool = False,
) -> None:
    """Cluster embeddings with cuML KMeans and optionally produce per-section scatter plots.

    Embeddings from all embeddings_dirs are concatenated before clustering.
    All k values share the same loaded embedding matrix (loaded once).
    Parquet metadata files must be co-located with their corresponding .npz files.

    Args:
        embeddings_dirs: One or more directories containing *_embeddings.npz files.
                         Defaults to /data2/alexlee/embeddings/638850.
                         Pass multiple: --embeddings-dirs /path/a --embeddings-dirs /path/b
        output_base:     Root directory for outputs (labels + plot dirs written here).
        tag:             Prefix for output filenames, e.g. '638850_687997'.
        ks:              List of cluster counts. Defaults to [100].
                         Pass multiple: --ks 100 --ks 1000
        n_gpus:          Number of GPUs for KMeans. 1 = single-GPU cuml.cluster.KMeans.
                         >1 = multi-GPU cuml.dask.cluster.KMeans via LocalCUDACluster.
                         Benchmark shows ~2.8–3.8x speedup at 5M+ cells with n_gpus=4.
        random_state:    RNG seed for KMeans and colour shuffling.
        s:               Scatter-plot marker size passed to make_section_plot.
        x_col:           Parquet column for x spatial coordinate.
        y_col:           Parquet column for y spatial coordinate.
        section_col:     Parquet column for tissue section identifier.
        dry_run:         Print summary stats and exit without clustering or plotting.
        no_plots:        Save labels only, skip per-section scatter plots.
    """
    if embeddings_dirs is None:
        embeddings_dirs = ["/data2/alexlee/embeddings/638850"]
    if ks is None:
        ks = [100]

    base_out = Path(output_base)

    # ------------------------------------------------------------------ #
    # 1. Load embeddings + parquet metadata (once for all k)
    # ------------------------------------------------------------------ #
    all_embeddings = []
    meta_rows = []

    for emb_dir_str in embeddings_dirs:
        emb_dir = Path(emb_dir_str)
        npz_paths = sorted(emb_dir.glob("*_embeddings.npz"))
        if not npz_paths:
            raise FileNotFoundError(f"No *_embeddings.npz files found in {emb_dir}")
        print(f"[{emb_dir.name}] {len(npz_paths)} section files")

        for npz_path in npz_paths:
            stem = npz_path.stem.removesuffix("_embeddings")
            parquet_path = npz_path.with_name(f"{stem}.parquet")
            if not parquet_path.exists():
                raise FileNotFoundError(f"Missing parquet: {parquet_path}")

            data = np.load(npz_path)
            emb = data["embeddings"].astype(np.float32)
            n_cells = emb.shape[0]

            meta = pd.read_parquet(parquet_path, columns=[x_col, y_col, section_col])
            if len(meta) != n_cells:
                raise ValueError(
                    f"{npz_path.name}: embedding rows ({n_cells}) != parquet rows ({len(meta)})"
                )

            all_embeddings.append(emb)
            meta_rows.append(meta)
            print(f"  {stem}: {n_cells:>8,} cells")

    embeddings = np.concatenate(all_embeddings, axis=0)
    meta_df = pd.concat(meta_rows, axis=0, ignore_index=True)

    print(f"\nTag: {tag}  |  Will cluster at k={ks}  |  n_gpus={n_gpus}")
    print(f"Total cells: {len(meta_df):,}  |  Embeddings shape: {embeddings.shape}")

    if dry_run:
        print("\n[dry-run] Stopping before clustering.")
        return

    # ------------------------------------------------------------------ #
    # 2. Cluster + visualise for each k
    # ------------------------------------------------------------------ #
    base_out.mkdir(parents=True, exist_ok=True)
    for k in ks:
        _cluster_and_plot(embeddings, meta_df, k, base_out, tag, random_state, s,
                          x_col=x_col, y_col=y_col, section_col=section_col,
                          n_gpus=n_gpus, no_plots=no_plots)

    print("\nAll done.")


def cli():
    typer.run(main)


if __name__ == "__main__":
    cli()
