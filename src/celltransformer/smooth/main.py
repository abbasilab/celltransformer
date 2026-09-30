"""
Gaussian spatial smoothing of cell embeddings.

Reads the per-section output of celltransformer-extract: every
{stem}_embeddings.npz in embeddings_dir with its co-located {stem}.parquet.
For each section independently, builds a k-nearest-neighbor graph in (x, y)
space using cuML's GPU-accelerated NearestNeighbors, then replaces each cell's
embedding with a Gaussian-weighted average of its k neighbors' embeddings.

The Gaussian kernel is applied to the Euclidean distances returned by the kNN
query: w_i = exp(-d_i^2 / (2 * sigma^2)), then row-normalized. The smoothed
embedding for cell j is sum_i(w_ij * embed_i).

Operates per-section to respect tissue boundaries (no cross-section smoothing).
Neighbor embeddings are gathered in chunks of cells to bound memory.

Output uses the same layout as the input, so celltransformer-cluster can read
it directly:
  {output_dir}/{stem}_embeddings.npz   embeddings (smoothed), has_neighbors
  {output_dir}/{stem}.parquet          copy of the input metadata

Requires cuML (RAPIDS) for GPU-accelerated kNN. Use the bfm_cml conda env.
"""

import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _gaussian_weights(distances: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian kernel on distance matrix, row-normalized."""
    weights = torch.exp(-distances**2 / (2 * sigma**2))
    return weights / weights.sum(dim=1, keepdim=True)


def main(
    embeddings_dir: str,
    output_dir: str,
    gaussian_kernel: float = 100,
    spatial_x_col: str = "x",
    spatial_y_col: str = "y",
    n_neighbors: int = 500,
    chunk_size: int = 1024,
) -> None:
    """Spatially smooth per-section embeddings via kNN + Gaussian kernel.

    Args:
        embeddings_dir: Folder of *_embeddings.npz files with co-located .parquet.
        output_dir: Where to write the smoothed .npz files and parquet copies.
                    Must differ from embeddings_dir (filenames are the same).
        gaussian_kernel: Gaussian sigma in spatial coordinate units.
        spatial_x_col: Parquet column for x coordinates.
        spatial_y_col: Parquet column for y coordinates.
        n_neighbors: Number of neighbors for kNN (capped at the section size).
        chunk_size: Cells per chunk when gathering neighbor embeddings.
    """
    import cuml

    in_dir = Path(embeddings_dir)
    out_dir = Path(output_dir)

    #refuse to write over extract's output
    if out_dir.resolve() == in_dir.resolve():
        raise ValueError("output_dir must differ from embeddings_dir")

    npz_paths = sorted(in_dir.glob("*_embeddings.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No *_embeddings.npz files found in {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Processing {len(npz_paths)} sections")

    for i, npz_path in enumerate(npz_paths):
        #pair with parquet (same rule as celltransformer-cluster)
        stem = npz_path.stem.removesuffix("_embeddings")
        parquet_path = npz_path.with_name(f"{stem}.parquet")
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing parquet: {parquet_path}")

        #load one section
        data = np.load(npz_path)
        embedding = torch.from_numpy(data["embeddings"].astype(np.float32))
        coords = pd.read_parquet(parquet_path, columns=[spatial_x_col, spatial_y_col]).values
        if len(coords) != len(embedding):
            raise ValueError(
                f"{npz_path.name}: embedding rows ({len(embedding)}) != parquet rows ({len(coords)})"
            )
        n_cells = len(coords)
        log.info(f"[{i + 1}/{len(npz_paths)}] {stem} ({n_cells} cells)")

        #kNN within this section, so indices are local to this section's embeddings
        k = min(n_neighbors, n_cells)
        knn = cuml.NearestNeighbors(n_neighbors=k)
        knn.fit(coords)
        dists, neighbors = knn.kneighbors(coords)
        dists = torch.from_numpy(np.asarray(dists))
        neighbors = torch.from_numpy(np.asarray(neighbors))

        #gaussian-weighted average, a chunk of cells at a time
        smoothed = torch.empty_like(embedding)
        for start in range(0, n_cells, chunk_size):
            stop = start + chunk_size
            weights = _gaussian_weights(dists[start:stop], gaussian_kernel)
            neighbor_embeds = embedding[neighbors[start:stop]]
            smoothed[start:stop] = (neighbor_embeds * weights.unsqueeze(-1)).sum(dim=1)

        #write in celltransformer-cluster's layout
        np.savez(
            out_dir / npz_path.name,
            embeddings=smoothed.numpy(),
            has_neighbors=data["has_neighbors"],
        )
        shutil.copy(parquet_path, out_dir / parquet_path.name)

    log.info(f"Saved smoothed embeddings to {out_dir}")


def cli():
    typer.run(main)


if __name__ == "__main__":
    cli()
