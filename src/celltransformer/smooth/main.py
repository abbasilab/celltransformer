"""
Gaussian spatial smoothing of cell embeddings.

For each tissue section independently, builds an approximate k-nearest-neighbor
graph in (x, y) space using cuML's GPU-accelerated NearestNeighbors, then
replaces each cell's embedding with a Gaussian-weighted average of its k
neighbors' embeddings.

The Gaussian kernel is applied to the Euclidean distances returned by the kNN
query: w_i = exp(-d_i^2 / (2 * sigma^2)), then row-normalized. The smoothed
embedding for cell j is sum_i(w_ij * embed_i).

Operates per-section to respect tissue boundaries (no cross-section smoothing).
Positional alignment between the metadata CSV and the embedding tensor is
assumed (row i in CSV == row i in the embedding tensor).

Requires cuML (RAPIDS) for GPU-accelerated kNN. Use the bfm_cml conda env.
"""

import logging
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
    metadata_path: str,
    embedding_path: str,
    output_path: str,
    gaussian_kernel: int = 100,
    spatial_x_col: str = "x",
    spatial_y_col: str = "y",
    section_col: str = "section",
    n_neighbors: int = 500,
) -> None:
    """Spatially smooth embeddings via kNN + Gaussian kernel, per section.

    Args:
        metadata_path: CSV with spatial coordinates and section labels.
        embedding_path: .pt file with cell embeddings (N x D tensor).
        output_path: Where to save the smoothed embeddings (.pt).
        gaussian_kernel: Gaussian sigma in spatial coordinate units.
        spatial_x_col: Column name for x coordinates.
        spatial_y_col: Column name for y coordinates.
        section_col: Column name for tissue section ID.
        n_neighbors: Number of neighbors for approximate kNN.
    """
    import cuml

    metadata = pd.read_csv(metadata_path).reset_index(drop=True)
    log.info(f"Loaded metadata: {len(metadata)} cells")

    embedding = torch.load(embedding_path, weights_only=True)
    log.info(f"Loaded embeddings: {embedding.shape}")

    if len(metadata) != len(embedding):
        raise ValueError(
            f"Length mismatch: metadata={len(metadata)}, embedding={len(embedding)}"
        )

    smoothed = torch.empty_like(embedding)
    sections = metadata[section_col].unique()
    log.info(f"Processing {len(sections)} sections")

    for i, (section, group) in enumerate(metadata.groupby(section_col)):
        log.info(f"[{i + 1}/{len(sections)}] {section} ({len(group)} cells)")

        idx = group.index.values
        coords = group[[spatial_x_col, spatial_y_col]].values

        knn = cuml.NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(coords)
        dists, neighbors = knn.kneighbors(coords)

        weights = _gaussian_weights(torch.from_numpy(dists), gaussian_kernel)
        neighbor_embeds = embedding[torch.from_numpy(neighbors)]
        smoothed[idx] = (neighbor_embeds * weights.unsqueeze(-1)).sum(dim=1)

    log.info(f"Saving smoothed embeddings to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(smoothed, output_path)


def cli():
    typer.run(main)


if __name__ == "__main__":
    cli()
