"""Entry point: celltransformer-preprocess / python -m celltransformer.preprocess

Split a multi-section h5ad+parquet pair into per-section files.

Each output section gets its own .h5ad and .parquet file, with positional
alignment preserved (row i in parquet == row i in adata.X).

If the metadata has fewer rows than the h5ad, adata is first subsetted to
the cells present in the metadata via obs_names (or a specified ID column).

If no paired .parquet/.csv is found, adata.obs is used directly as metadata.

Optionally transforms .layers['raw'] → .X using log1p or log2(1+x).
Optionally filters to cells where a boolean obs column is True (--qc-col).

Functions:
  Transform             — Enum for expression transforms (log1p, log2p1)
  find_metadata         — Find paired .parquet or .csv alongside an .h5ad
  load_metadata         — Load metadata from .parquet or .csv
  subset_adata_to_metadata — Subset and align adata to metadata rows
  apply_transform       — Apply log transform to .layers['raw'] → .X
  split_sections        — Split multi-section adata into per-section files
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import anndata
import numpy as np
import pandas as pd
import typer
from scipy import sparse


class Transform(str, Enum):
    log1p  = "log1p"   # ln(1 + x)
    log2p1 = "log2p1"  # log2(1 + x)


# ---------------------------------------------------------------------------
# Metadata discovery
# ---------------------------------------------------------------------------

def find_metadata(h5ad_path: Path) -> Path:
    """Return the paired .parquet or .csv metadata file, or raise."""
    for suffix in (".parquet", ".csv"):
        candidate = h5ad_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No paired .parquet or .csv found next to {h5ad_path}"
    )


def load_metadata(meta_path: Path) -> pd.DataFrame:
    if meta_path.suffix == ".parquet":
        df = pd.read_parquet(meta_path)
    else:
        df = pd.read_csv(meta_path)
    return df.reset_index(drop=True)


# Column names tried in order when --obs-id-col is not specified
_DEFAULT_ID_COLS = ("cell_label", "production_cell_id")


# ---------------------------------------------------------------------------
# Subset adata to cells present in metadata
# ---------------------------------------------------------------------------

def subset_adata_to_metadata(
    adata: anndata.AnnData,
    metadata: pd.DataFrame,
    obs_id_col: str | None,
) -> tuple[anndata.AnnData, pd.DataFrame]:
    """Subset adata to the cells listed in metadata, keyed by obs_names.

    obs_id_col: column in metadata whose values match adata.obs_names.
                If None, tries 'cell_label' then 'production_cell_id' in order,
                then falls back to the metadata integer index.

    Returns (subsetted_adata, reordered_metadata) with guaranteed positional
    alignment between the two.
    """
    if obs_id_col is not None:
        if obs_id_col not in metadata.columns:
            raise typer.BadParameter(
                f"--obs-id-col '{obs_id_col}' not found in metadata. "
                f"Available: {list(metadata.columns)}"
            )
        cell_ids = metadata[obs_id_col].astype(str).values
        typer.echo(f"  using specified ID column: '{obs_id_col}'")
    else:
        for candidate in _DEFAULT_ID_COLS:
            if candidate in metadata.columns:
                cell_ids = metadata[candidate].astype(str).values
                typer.echo(f"  auto-detected ID column: '{candidate}'")
                break
        else:
            cell_ids = metadata.index.astype(str).values
            typer.echo("  no ID column found; using metadata integer index as cell IDs")

    obs_names = adata.obs_names.astype(str)

    ids_in_adata = pd.Index(cell_ids).intersection(obs_names)
    n_meta   = len(cell_ids)
    n_common = len(ids_in_adata)
    n_adata  = adata.n_obs

    typer.echo(
        f"  adata obs_names : {n_adata:>10,}\n"
        f"  metadata rows   : {n_meta:>10,}\n"
        f"  common cells    : {n_common:>10,}"
    )

    if n_common == 0:
        typer.secho(
            "ERROR: no common cell IDs found between adata.obs_names and metadata. "
            "Check --obs-id-col or verify file pairing.",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(1)

    dropped = n_adata - n_common
    if dropped:
        typer.secho(
            f"  → dropping {dropped:,} cells from adata not present in metadata",
            fg=typer.colors.YELLOW,
        )

    obs_pos = pd.Series(np.arange(n_adata), index=obs_names)

    keep_mask   = pd.Index(cell_ids).isin(obs_names)
    meta_kept   = metadata[keep_mask].reset_index(drop=True)
    kept_ids    = cell_ids[keep_mask]

    adata_indices = obs_pos[kept_ids].values
    adata_sub     = adata[adata_indices].copy()

    return adata_sub, meta_kept


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def apply_transform(section_adata: anndata.AnnData, transform: Transform) -> None:
    """Transform .layers['raw'] → .X in-place on a section AnnData."""
    if "raw" not in section_adata.layers:
        raise KeyError(
            "'.layers[\"raw\"]' not found in AnnData. "
            "Cannot apply transform without raw counts."
        )
    raw = section_adata.layers["raw"]
    if sparse.issparse(raw):
        raw = raw.toarray()
    else:
        raw = np.asarray(raw, dtype=np.float32)

    if transform == Transform.log1p:
        section_adata.X = np.log1p(raw).astype(np.float32)
    elif transform == Transform.log2p1:
        section_adata.X = np.log2(raw + 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------

def split_sections(
    adata: anndata.AnnData,
    metadata: pd.DataFrame,
    section_col: str,
    output_dir: Path,
    stem: str,
    transform: Transform | None,
    dry_run: bool,
) -> None:
    if section_col not in metadata.columns:
        raise typer.BadParameter(
            f"Section column '{section_col}' not found in metadata. "
            f"Available columns: {list(metadata.columns)}"
        )

    nan_rows = metadata[section_col].isna().sum()
    if nan_rows:
        typer.secho(
            f"  Warning: dropping {nan_rows:,} rows with NaN in '{section_col}'.",
            fg=typer.colors.YELLOW,
        )
        keep = metadata[section_col].notna()
        metadata = metadata[keep].reset_index(drop=True)
        adata = adata[keep.values].copy()

    sections = sorted(metadata[section_col].unique())
    typer.echo(f"Found {len(sections)} section(s): {sections}\n")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for sec in sections:
        mask = (metadata[section_col] == sec).values

        section_adata = adata[mask].copy()
        section_meta  = metadata[mask].reset_index(drop=True)

        n_cells  = int(mask.sum())
        sec_safe = str(sec).replace(".", "_").replace(" ", "_")
        out_h5ad = output_dir / f"{stem}_{sec_safe}.h5ad"
        out_pq   = output_dir / f"{stem}_{sec_safe}.parquet"

        typer.echo(f"  [{sec}]  {n_cells:>8,} cells")
        typer.echo(f"           h5ad    → {out_h5ad}")
        typer.echo(f"           parquet → {out_pq}")

        if dry_run:
            typer.echo("")
            continue

        if transform is not None:
            apply_transform(section_adata, transform)

        section_adata.write_h5ad(out_h5ad)
        section_meta.to_parquet(out_pq, index=False)
        typer.echo("")

    if dry_run:
        typer.secho("[dry-run] No files written.", fg=typer.colors.YELLOW)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_h5ad: Path = typer.Argument(
        ...,
        help="Path to multi-section .h5ad file.",
        exists=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o",
        help="Directory where per-section output pairs are written.",
    ),
    metadata: Optional[Path] = typer.Option(
        None, "--metadata", "-m",
        help="Explicit path to paired metadata file (.parquet or .csv). "
             "Defaults to auto-discovery (same stem as h5ad).",
        exists=True,
        dir_okay=False,
    ),
    section_col: str = typer.Option(
        "section",
        help="Column in the metadata file that identifies tissue sections.",
    ),
    obs_id_col: Optional[str] = typer.Option(
        None, "--obs-id-col",
        help="Column in metadata containing cell IDs matching adata.obs_names. "
             "Used when row counts differ (metadata is a subset of adata). "
             "Auto-tries 'cell_label' then 'production_cell_id' before falling back to index.",
    ),
    transform: Optional[Transform] = typer.Option(
        None,
        help="Transform .layers['raw'] into .X before saving. "
             "log1p=ln(1+x), log2p1=log2(1+x).",
    ),
    output_stem: Optional[str] = typer.Option(
        None,
        help="Base name for output files ({stem}_{section}.h5ad). "
             "Defaults to input file stem.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print section names and cell counts without writing any files.",
    ),
    qc_col: Optional[str] = typer.Option(
        None, "--qc-col",
        help="Boolean column in adata.obs to filter cells on (keeps True rows). "
             "E.g. 'scvi_qc_passed_sup_level'. Defaults to None (no filter).",
    ),
) -> None:
    stem = output_stem or input_h5ad.stem

    typer.echo(f"Loading {input_h5ad} ...")
    adata = anndata.read_h5ad(input_h5ad)
    typer.echo(f"  {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    if metadata is not None:
        meta_path = metadata
        typer.echo(f"Loading metadata from {meta_path} ...")
        meta_df = load_metadata(meta_path)
        typer.echo(f"  {len(meta_df):,} rows")
    else:
        try:
            meta_path = find_metadata(input_h5ad)
            typer.echo(f"Loading metadata from {meta_path} ...")
            meta_df = load_metadata(meta_path)
            typer.echo(f"  {len(meta_df):,} rows")
        except FileNotFoundError:
            typer.secho(
                "No paired .parquet/.csv found — using adata.obs as metadata.",
                fg=typer.colors.YELLOW,
            )
            meta_df = adata.obs.reset_index(drop=True)
            typer.echo(f"  {len(meta_df):,} rows (from obs)")

    if len(meta_df) != adata.n_obs:
        typer.secho(
            f"\nRow count mismatch ({adata.n_obs:,} vs {len(meta_df):,}) — "
            "subsetting adata to cells present in metadata via obs_names ...",
            fg=typer.colors.YELLOW,
        )
        adata, meta_df = subset_adata_to_metadata(adata, meta_df, obs_id_col)
        typer.echo(f"  → aligned: {adata.n_obs:,} cells")

    if qc_col is not None:
        if qc_col not in adata.obs.columns:
            typer.secho(
                f"ERROR: --qc-col '{qc_col}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}",
                fg=typer.colors.RED, err=True,
            )
            raise typer.Exit(1)
        qc_mask = adata.obs[qc_col].astype(bool).values
        n_before = adata.n_obs
        n_kept = int(qc_mask.sum())
        n_filtered = n_before - n_kept
        adata   = adata[qc_mask].copy()
        meta_df = meta_df[qc_mask].reset_index(drop=True)
        typer.echo(
            f"\nQC filter '{qc_col}':\n"
            f"  before   : {n_before:>10,}\n"
            f"  kept     : {n_kept:>10,}  ({100 * n_kept / n_before:.1f}%)\n"
            f"  filtered : {n_filtered:>10,}  ({100 * n_filtered / n_before:.1f}%)"
        )

    blank_mask = adata.var_names.str.contains("blank", case=False)
    n_blanks = blank_mask.sum()
    if n_blanks:
        typer.echo(
            f"\nRemoving {n_blanks} blank gene(s) from {adata.n_vars} total "
            f"→ {adata.n_vars - n_blanks} genes kept."
        )
        adata = adata[:, ~blank_mask].copy()

    typer.echo("")
    split_sections(
        adata=adata,
        metadata=meta_df,
        section_col=section_col,
        output_dir=output_dir,
        stem=stem,
        transform=transform,
        dry_run=dry_run,
    )


cli = app

if __name__ == "__main__":
    app()
