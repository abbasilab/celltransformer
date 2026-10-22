"""Entry point: celltransformer-extract / python -m celltransformer.extract

Extract spatial-context embeddings from a trained SpatialCellTransformer checkpoint.

For each center cell the spatial neighbourhood is extracted and passed through:
  CellEmbedder → TransformerEncoder stack → AttentionPool

The AttentionPool compresses the variable-length neighbourhood into a single
spatial_context vector saved as embeddings.

Cells with zero neighbours receive a zero embedding and are flagged via the
`has_neighbors` boolean array in the output.

Output (one .npz per input .h5ad):
  {output_dir}/{stem}_embeddings.npz
    embeddings    : float32  (n_cells, model_dim)
    has_neighbors : bool     (n_cells,)

The script is resumable: if the output .npz already exists it is skipped.
Works for both single-section and multi-section .h5ad files.

Example:
    celltransformer-extract data.h5ad \\
        --checkpoint-path /path/to/last.ckpt \\
        --cell-type-yaml /path/to/taxonomy.yaml \\
        --cell-type-colname hrc_mmc_subclass_name

If the model was trained with model.use_cell_type=false (read from the
checkpoint), --cell-type-yaml and --cell-type-colname are not needed.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
import typer
from omegaconf import OmegaConf

from celltransformer.model import SpatialCellTransformer
from celltransformer.loader import CenterMaskNodeOptimized


def _build_cell_type_map(yaml_path: str) -> dict:
    with open(yaml_path) as f:
        labels = yaml.safe_load(f)
    return {label: idx for idx, label in enumerate(labels)}


def _load_model(ckpt_path: str, device: str) -> SpatialCellTransformer:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.") # current lightning code will store the model as a model dotattr
    }
    cfg = OmegaConf.to_container(ckpt["hyper_parameters"]["model_cfg"], resolve=True)
    cfg.pop("_target_", None)
    cfg.pop("compile", None)
    model = SpatialCellTransformer(**cfg, device="cpu", dtype=torch.float32)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


@torch.no_grad()
def _embed(
    model: SpatialCellTransformer,
    dataset: CenterMaskNodeOptimized,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells   = len(dataset)
    model_dim = model.attn_pool.output_dim
    embeddings    = np.zeros((n_cells, model_dim), dtype=np.float32)
    has_neighbors = np.zeros(n_cells, dtype=bool)

    for start in range(0, n_cells, batch_size):
        end   = min(start + batch_size, n_cells)
        batch = [dataset[i] for i in range(start, end)]

        valid_local = [i for i, el in enumerate(batch) if el.num_cells_obs > 0]
        if not valid_local:
            continue

        items = [batch[i] for i in valid_local]
        lengths        = torch.LongTensor([el.num_cells_obs for el in items]).to(device)
        expression_mat = torch.cat([el.observed_expression for el in items], 0).float().to(device)
        cell_type_obs  = torch.cat([el.observed_cell_type   for el in items], 0).to(device)
        decoder_ct     = torch.cat([el.masked_cell_type     for el in items], 0).to(device)

        out = model(
            expression_mat=expression_mat,
            cell_type_annot_obs=cell_type_obs,
            decoder_cell_type_annot=decoder_ct,
            lengths=lengths,
        )
        sc = out["spatial_context"].values().cpu().float().numpy()

        for batch_pos, local_i in enumerate(valid_local):
            embeddings[start + local_i]    = sc[batch_pos]
            has_neighbors[start + local_i] = True

    return embeddings, has_neighbors


def main(
    h5ad_path: str,
    checkpoint_path: str = typer.Option(..., help="Path to Lightning .ckpt file."),
    cell_type_yaml: Optional[str] = typer.Option(None, help="Path to taxonomy YAML (ordered list of label strings). Required unless the model was trained with model.use_cell_type=false."),
    cell_type_colname: Optional[str] = typer.Option(None, help="Column in parquet with cell type label strings. Required unless the model was trained with model.use_cell_type=false."),
    section_colname: str = typer.Option("section", help="Column in parquet with tissue section IDs."),
    patch_size: float = typer.Option(31, help="Square spatial window size in coordinate units."),
    output_dir: str = typer.Option("", help="Directory to write .npz. Defaults to same dir as h5ad."),
    device: str = typer.Option("cuda:0", help="CUDA device string, e.g. cuda:0 or cuda:1."),
    batch_size: int = typer.Option(512, help="Cells per GPU forward pass."),
    dry_run: bool = typer.Option(False, help="Print what would be processed and exit."),
) -> None:
    """Embed all cells in a single .h5ad using a trained SpatialCellTransformer.

    Writes {output_dir}/{stem}_embeddings.npz with arrays:
      embeddings    : float32 (n_cells, model_dim)
      has_neighbors : bool    (n_cells,)

    The .npz row order matches the .h5ad / .parquet row order exactly.
    """
    h5ad = Path(h5ad_path)
    pq   = h5ad.with_suffix(".parquet")
    if not h5ad.exists():
        raise FileNotFoundError(h5ad)
    if not pq.exists():
        raise FileNotFoundError(f"No paired parquet found: {pq}")

    out_dir  = Path(output_dir) if output_dir else h5ad.parent
    out_file = out_dir / f"{h5ad.stem}_embeddings.npz"

    print(f"Input : {h5ad.name}")
    print(f"Output: {out_file}")

    if dry_run:
        print("[dry-run] Exiting.")
        return

    if out_file.exists():
        print("Already done — skipping.")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    model = _load_model(checkpoint_path, device)
    print(f"Model ready — model_dim={model.attn_pool.output_dim}, device={device}")

    # the checkpoint records whether training used cell types; extraction must match
    if model.use_cell_type:
        if cell_type_yaml is None or cell_type_colname is None:
            raise typer.BadParameter(
                "This model was trained with cell types: pass --cell-type-yaml and "
                "--cell-type-colname (the same taxonomy and column used for training)."
            )
        cell_type_map = _build_cell_type_map(cell_type_yaml)
    else:
        if cell_type_yaml is not None or cell_type_colname is not None:
            print("Model was trained with use_cell_type=false — ignoring cell-type options.")
        cell_type_map = None

    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building dataset...", end="", flush=True)
    dataset = CenterMaskNodeOptimized(
        metadata_path=str(pq),
        adata_path=str(h5ad),
        cell_type_colname=cell_type_colname,
        patch_size=(patch_size, patch_size),
        tissue_section_colname=section_colname,
        cell_type_int_map=cell_type_map,
        use_cell_type=model.use_cell_type,
    )
    n_cells = len(dataset)
    print(f" {n_cells:,} cells — embedding...", end="", flush=True)

    embeddings, has_neighbors = _embed(model, dataset, batch_size, device)

    np.savez(out_file, embeddings=embeddings, has_neighbors=has_neighbors)
    n_valid = int(has_neighbors.sum())
    print(f" done ({n_valid:,}/{n_cells:,} with neighbours) → {out_file.name}")


def cli():
    typer.run(main)


if __name__ == "__main__":
    cli()
