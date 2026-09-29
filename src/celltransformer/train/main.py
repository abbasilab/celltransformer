"""Entry point: celltransformer-train / python -m celltransformer.train

Launches training via Hydra. Config is read from the sibling config/ directory
at the repo root. Override any config value on the command line:

    celltransformer-train data=scvi_sections model.dropout=0.1
    python -m celltransformer.train data=scvi_sections model.dropout=0.1

Exports:
  SpatialTransformerModule  — Lightning module wrapping SpatialCellTransformer
  zinb_loss_log2            — ZINB NLL loss on log2-transformed counts
  build_cell_type_int_map   — Build {label: int} mapping from YAML or LabelEncoder
"""

import json
import warnings
warnings.filterwarnings("ignore", message="The value argument must be within the support of the distribution")
warnings.filterwarnings("ignore", message="Importing CSCDataset from `anndata.experimental` is deprecated", category=FutureWarning)
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import wandb
import yaml
import lightning as L
import hydra
import pandas as pd
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import LabelEncoder

from celltransformer.loader import MERFISHDataModule
from celltransformer.train import scheduler
from celltransformer.distributions import ZeroInflatedNegativeBinomial

torch.set_float32_matmul_precision('high')


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def build_cell_type_int_map(
    cell_type_colname: str,
    data_dir: str,
    patterns: list[str],
    cell_type_yaml: Optional[str],
) -> dict[str, int]:
    """Return a {label_string: int} mapping built from either a taxonomy YAML
    or LabelEncoder.fit_transform over all matching parquet files.

    Taxonomy YAML (preferred): ordered list of label strings; integer = index.
    LabelEncoder fallback: scans all section parquets, fits on the union of
    unique values. Consistent within a run but not across runs.
    """
    if cell_type_yaml is not None:
        with open(cell_type_yaml) as f:
            labels: list[str] = yaml.safe_load(f)
        mapping = {label: idx for idx, label in enumerate(labels)}
        print(f"[label_enc] taxonomy YAML → {len(mapping)} classes  ({cell_type_yaml})", flush=True)
        return mapping

    # LabelEncoder fallback: collect all unique values from parquet files
    data_path = Path(data_dir)
    all_values: set[str] = set()
    for pattern in patterns:
        for pq_path in sorted(data_path.glob(pattern.replace("h5ad", "parquet"))):
            col = pd.read_parquet(pq_path, columns=[cell_type_colname])[cell_type_colname]
            all_values.update(col.dropna().astype(str).unique())

    le = LabelEncoder()
    le.fit(sorted(all_values))
    mapping = {label: int(idx) for idx, label in enumerate(le.classes_)}
    print(f"[label_enc] LabelEncoder fallback → {len(mapping)} classes", flush=True)
    return mapping


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class SpatialTransformerModule(L.LightningModule):
    def __init__(self, model_cfg: DictConfig, opt_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.opt_cfg = opt_cfg
        self.model = instantiate(model_cfg, device='cpu', dtype=torch.float32)

        self._train_cells_epoch = 0
        self._epoch_start_time = None
        self._epoch_time = None
        self._total_cells = 0

    def on_train_epoch_start(self):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        self._epoch_start_time = time.time()

    def _prepare_batch(self, batch_split):
        """Filter zero-neighbor cells, concatenate fields, move to device."""
        batch_split = [el for el in batch_split if el.num_cells_obs > 0]
        if not batch_split:
            return None
        dev = self.device
        return dict(
            lengths=torch.LongTensor([el.num_cells_obs for el in batch_split]).to(dev, non_blocking=True),
            expression_mat=torch.cat([el.observed_expression for el in batch_split], dim=0).to(dev, non_blocking=True),
            cell_type_annot_obs=torch.cat([el.observed_cell_type for el in batch_split], dim=0).to(dev, non_blocking=True),
            decoder_cell_type_annot=torch.cat([el.masked_cell_type for el in batch_split], dim=0).to(dev, non_blocking=True),
            gt_expr=torch.cat([el.masked_expression for el in batch_split]).to(dev, non_blocking=True),
        )

    def training_step(self, batch_split):
        batch = self._prepare_batch(batch_split)
        if batch is None:
            return None

        out = self.model(
            expression_mat=batch['expression_mat'],
            cell_type_annot_obs=batch['cell_type_annot_obs'],
            decoder_cell_type_annot=batch['decoder_cell_type_annot'],
            lengths=batch['lengths'],
        )

        zinb    = ZeroInflatedNegativeBinomial(**out['nb_params'])
        gt_expr = batch['gt_expr']
        loss, _ = zinb_loss_log2(zinb, gt_expr)
        self._train_cells_epoch += int(gt_expr.shape[0])
        batch_size = gt_expr.shape[0]
        self.log("train/nll", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        return loss

    def on_train_epoch_end(self):
        elapsed = time.time() - self._epoch_start_time
        total_cells = torch.tensor(self._train_cells_epoch, device=self.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(total_cells)
        self._epoch_time  = elapsed
        self._total_cells = int(total_cells.item())
        if self.trainer.global_rank == 0:
            cells_per_sec = self._total_cells / elapsed if elapsed > 0 else 0
            print(f"[train] epoch {self.current_epoch} | {elapsed:.1f}s | {self._total_cells} cells | {cells_per_sec:.0f} cells/s", flush=True)
            self.log("train/cells_per_sec", cells_per_sec, rank_zero_only=True)
            self.log("train/total_cells",   float(self._total_cells), rank_zero_only=True)
        self._train_cells_epoch = 0

    def validation_step(self, batch_split, batch_idx):
        batch = self._prepare_batch(batch_split)
        if batch is None:
            return

        out = self.model(
            expression_mat=batch['expression_mat'],
            cell_type_annot_obs=batch['cell_type_annot_obs'],
            decoder_cell_type_annot=batch['decoder_cell_type_annot'],
            lengths=batch['lengths'],
        )

        zinb    = ZeroInflatedNegativeBinomial(**out['nb_params'])
        gt_expr = batch['gt_expr']
        loss, _ = zinb_loss_log2(zinb, gt_expr)
        batch_size = gt_expr.shape[0]
        self.log("valid/nll", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True, batch_size=batch_size)

    def configure_optimizers(self):
        cfg = self.opt_cfg
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        lr_sched = scheduler.get_inverse_sqrt_schedule_with_plateau(
            optimizer,
            num_warmup_steps=cfg.schedule_warmup,
            timescale=cfg.schedule_timescale,
        )
        return [optimizer], dict(scheduler=lr_sched, frequency=1, interval="step")


def zinb_loss_log2(zinb_obj, gt_expr: torch.Tensor):
    gt_expr_counts = torch.pow(2, gt_expr) - 1.
    return -zinb_obj.log_prob(gt_expr_counts).mean(), gt_expr_counts


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

# Resolve config/ directory relative to repo root.
# src/celltransformer/train/main.py → parents[3] = repo root
_CONFIG_DIR = str(Path(__file__).parents[3] / "config")


@hydra.main(config_path=_CONFIG_DIR, config_name="train_redone_registration", version_base=None)
def main(cfg: DictConfig) -> None:
    d = cfg.data
    o = cfg.optimization

    cell_type_int_map = build_cell_type_int_map(
        cell_type_colname=d.cell_type_colname,
        data_dir=d.data_dir,
        patterns=OmegaConf.to_container(d.train_anndata_patterns),
        cell_type_yaml=d.get("cell_type_yaml", None),
    )

    # --- Timestamped checkpoint directory ---------------------------------
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir     = Path(to_absolute_path(cfg.checkpoint_dir)) / timestamp
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "wandb").mkdir(parents=True, exist_ok=True)

    # --- W&B logger -------------------------------------------------------
    wandb_config = wandb.helper.parse_config({
        **OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
        "checkpoint_dir": str(ckpt_dir),
    })
    wandb_tags = list(OmegaConf.to_container(cfg.get("wandb_tags", [])) or [])
    wandb_notes = cfg.get("wandb_notes", "") or None
    wandb_logger = L.pytorch.loggers.WandbLogger(
        project=cfg.wandb_project,
        name=timestamp,
        save_dir=str(ckpt_dir),
        config=wandb_config,
        tags=wandb_tags or None,
        notes=wandb_notes,
    )

    # --- Checkpoint callback ----------------------------------------------
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch{epoch:03d}-step{step}-valloss{valid/nll_epoch:.4f}",
        monitor="valid/nll_epoch",
        mode="min",
        save_top_k=cfg.get("save_top_k", 5),
        save_last=True,
        every_n_epochs=1,
    )

    # --- Data module ------------------------------------------------------
    data_module = MERFISHDataModule(
        data_dir=d.data_dir,
        train_anndata_patterns=OmegaConf.to_container(d.train_anndata_patterns),
        valid_anndata_patterns=OmegaConf.to_container(d.valid_anndata_patterns),
        batch_size=d.batch_size,
        patch_size=tuple(d.patch_size),
        workers_per_split=d.workers_per_split,
        num_splits=d.num_splits,
        prefetch_factor=d.prefetch_factor,
        max_num_cells=None,
        max_tokens=d.get("max_tokens", None),
        multiprocessing_method=d.multiprocessing_method,
        read_mode=None,
        cell_type_colname=d.cell_type_colname,
        tissue_section_colname=d.tissue_section_colname,
        cell_type_int_map=cell_type_int_map,
        partition_strategy=d.partition_strategy,
    )

    lightning_model = SpatialTransformerModule(
        model_cfg=cfg.model,
        opt_cfg=o,
    )

    #GPU indices from the config (a list, or an int count)
    devices = OmegaConf.to_container(cfg, resolve=True)["devices"]
    trainer = L.Trainer(
        log_every_n_steps=100,
        max_epochs=o.max_epochs,
        precision=o.precision,
        accelerator="cuda",
        devices=devices,
        strategy=L.pytorch.strategies.DDPStrategy(find_unused_parameters=False, static_graph=True),
        accumulate_grad_batches=o.accumulate_grad_batches,
        deterministic=True,
        enable_checkpointing=True,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    ckpt_path = cfg.get("model_checkpoint", None)
    if ckpt_path is not None:
        ckpt_path = Path(to_absolute_path(str(ckpt_path)))
        if ckpt_path.is_dir():
            last = ckpt_path / "last.ckpt"
            if last.exists():
                ckpt_path = last
            else:
                candidates = sorted(ckpt_path.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
                ckpt_path = candidates[-1] if candidates else None
    if ckpt_path is not None:
        print(f"[train] resuming from checkpoint: {ckpt_path}", flush=True)
    else:
        print("[train] starting from scratch", flush=True)

    trainer.fit(
        lightning_model,
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )

    # --- Post-fit: persist W&B run metadata alongside checkpoints ---------
    if trainer.global_rank == 0:
        run = wandb_logger.experiment
        run_info = {
            "run_id":         run.id,
            "run_name":       run.name,
            "run_url":        run.url,
            "checkpoint_dir": str(ckpt_dir),
            "timestamp":      timestamp,
        }
        with open(ckpt_dir / "wandb_run.json", "w") as f:
            json.dump(run_info, f, indent=2)
        print(f"[train] W&B run:    {run.url}", flush=True)
        print(f"[train] checkpoints: {ckpt_dir}", flush=True)

        result = {
            "workers_per_split":       d.workers_per_split,
            "num_splits":              d.num_splits,
            "prefetch_factor":         d.prefetch_factor,
            "accumulate_grad_batches": o.accumulate_grad_batches,
            "epoch_time_s":            lightning_model._epoch_time,
            "total_cells":             lightning_model._total_cells,
            "cells_per_sec": (
                lightning_model._total_cells / lightning_model._epoch_time
                if lightning_model._epoch_time and lightning_model._epoch_time > 0 else 0
            ),
            "wandb_run_id":    run.id,
            "wandb_run_url":   run.url,
            "checkpoint_dir":  str(ckpt_dir),
        }
        out_path = cfg.get("output", "results.jsonl")
        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"[train] result written to {out_path}: {result}", flush=True)


if __name__ == "__main__":
    main()
