import sys
import logging
from os import PathLike
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import lightning as L
import torch
from torchdata.nodes import (
    MultiNodeWeightedSampler,
    Prefetcher,
    Loader,
)

from celltransformer.loader.distributed import _get_cell_count
from celltransformer.loader.base_node import AnnDataMetadataNode
from celltransformer.loader.sampler import CellParallelSampler, BudgetBatcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def collate(batch):
    obs_expr_list = []
    obs_cell_type_list = []
    masked_cell_type_list = []
    masked_expr_list = []
    lengths_list = []

    for el in batch:
        if el.num_cells_obs == 0:
            continue

        obs_expr_list.append(el.observed_expression)
        obs_cell_type_list.append(el.observed_cell_type)
        masked_cell_type_list.append(el.masked_cell_type)
        masked_expr_list.append(el.masked_expression)
        lengths_list.append(el.num_cells_obs)

    lengths = torch.LongTensor(lengths_list)
    expression_mat = torch.cat(obs_expr_list, dim=0)
    cell_type_annot_obs = torch.cat(obs_cell_type_list, dim=0)
    decoder_cell_type_annot = torch.cat(masked_cell_type_list, dim=0)
    gt_expr = torch.cat(masked_expr_list, dim=0)

    return dict(
        lengths=lengths,
        expression_mat=expression_mat,
        cell_type_annot_obs=cell_type_annot_obs,
        decoder_cell_type_annot=decoder_cell_type_annot,
        tot_len=lengths.sum(),
        gt_expr=gt_expr,
    )

class _LenLoader:
    """Thin wrapper that gives a torchdata Loader a fixed __len__.

    Lightning reads len(train_dataloader) to determine num_training_batches.
    When all ranks return the same length (the all_reduce minimum), Lightning
    stops every rank at the same step, preventing the DDP deadlock that arises
    when ranks exhaust their data at different wall-clock times.
    """
    def __init__(self, loader, length: int):
        self._loader = loader
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        return iter(self._loader)


class MERFISHDataModule(L.LightningDataModule):
    def __init__(self,
                 data_dir: PathLike,
                 batch_size: int,
                 num_splits: int,
                 train_anndata_patterns: List[str] = ["*.h5ad"],
                 valid_anndata_patterns: List[str] = ["*.h5ad"],
                 cell_type_colname: Optional[str] = "subclass_int",
                 patch_size: Union[list[int, int], tuple[int, int]] = [23, 23],
                 tissue_section_colname: Optional[str] = "section",
                 max_num_cells: Optional[int] = None,
                 max_tokens: Optional[int] = None,
                 multiprocessing_method: Optional[Literal['thread', 'process']] = 'thread',
                 workers_per_split: Optional[int] = 4,
                 prefetch_factor: Optional[int] = 3,
                 shuffle: Optional[bool] = True,
                 read_mode: Optional[str] = None,
                 cell_type_int_map: Optional[Dict[str, int]] = None,
                 partition_strategy: Literal["binpack", "roundrobin"] = "binpack",
                 ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_splits = num_splits
        self.train_anndata_patterns = train_anndata_patterns
        self.valid_anndata_patterns = valid_anndata_patterns
        self.cell_type_colname = cell_type_colname
        self.cell_type_int_map = cell_type_int_map
        self.patch_size = patch_size
        self.tissue_section_colname = tissue_section_colname
        self.max_num_cells = max_num_cells
        self.max_tokens = max_tokens
        self.multiprocessing_method = multiprocessing_method
        self.workers_per_split = workers_per_split
        self.prefetch_factor = prefetch_factor
        self.shuffle = shuffle
        self.mode = read_mode
        self.partition_strategy = partition_strategy

    def setup_node(self, anndata_patterns: List[str],
                   exclude_patterns: List[str],
                   shuffle: bool,
                   cycle: bool = False):
        node = AnnDataMetadataNode(
            root_dir=self.data_dir,
            anndata_patterns=anndata_patterns,
            exclude_patterns=exclude_patterns,
            max_num_files=None,
            partition_strategy=self.partition_strategy,
        )
        max_total_cells = node.max_total_cells
        subsample_info = node.subsample_info

        split_nodes = node.split(self.num_splits)
        # split() may return fewer nodes than self.num_splits if there are fewer
        # files than requested splits. Use the actual count as the divisor below.
        n_actual_splits = len(split_nodes)

        if n_actual_splits == 0:
            raise RuntimeError(
                f"setup_node: 0 file splits found for patterns {anndata_patterns!r} "
                f"in '{self.data_dir}'. This rank has no files after DDP sharding. "
                f"Reduce the number of GPUs or add more files to match the number of "
                f"GPUs × num_splits ({self.num_splits})."
            )

        weight_mapper = {key: 1. / n_actual_splits for key in split_nodes.keys()}

        # When cycle=True (training), no per-split cell cap. CYCLE_FOREVER
        # ensures the pipeline never raises StopIteration; Lightning's step
        # limit (_train_batches) stops all ranks at the same step.
        # When cycle=False (validation), use ALL_DATASETS_EXHAUSTED so each
        # cell is evaluated exactly once (clean metrics).
        if cycle:
            per_split_max_cells = None
            stop_criteria = 'CYCLE_FOREVER'
        else:
            if max_total_cells is not None:
                per_split_max_cells = max_total_cells // n_actual_splits
            else:
                per_split_max_cells = None
            stop_criteria = 'ALL_DATASETS_EXHAUSTED'

        mapped_nodes = {}
        for key in split_nodes.keys():
            this_node = split_nodes[key]
            mapped_nodes[key] = CellParallelSampler(
                this_node,
                self.workers_per_split,
                cell_type_colname = self.cell_type_colname,
                patch_size=self.patch_size,
                tissue_section_colname = self.tissue_section_colname,
                use_gene_mask = True,
                max_num_cells = self.max_num_cells,
                max_total_cells = per_split_max_cells,
                subsample_info = subsample_info,
                shuffle=shuffle,
                method = self.multiprocessing_method,
                dist_log_str=this_node.dist_log_str,
                read_mode=self.mode,
                cell_type_int_map=self.cell_type_int_map,
        )

        mn_sampler = MultiNodeWeightedSampler(
            mapped_nodes,
            weight_mapper,
            stop_criteria=stop_criteria,
        )

        prefetcher = Prefetcher(
            mn_sampler,
            prefetch_factor=self.prefetch_factor,
        )

        batcher = BudgetBatcher(
            prefetcher,
            max_batch_size=self.batch_size,
            max_tokens=self.max_tokens,
         )

        # Total cells this rank has across all splits.
        if cycle:
            # No capping — raw cell count for MAX all_reduce.
            rank_cells = sum(
                sum(_get_cell_count(meta_path) for (_, meta_path) in node._file_pairs)
                for node in split_nodes.values()
            )
        else:
            # Cap per split for MIN all_reduce (original validation behavior).
            _cap = per_split_max_cells if per_split_max_cells is not None else 10 ** 18
            rank_cells = sum(
                min(
                    sum(_get_cell_count(meta_path) for (_, meta_path) in node._file_pairs),
                    _cap,
                )
                for node in split_nodes.values()
            )

        return batcher, rank_cells

    def setup(self, stage: Optional[str] = None):
        train_node, train_cells = self.setup_node(
            self.train_anndata_patterns,
            exclude_patterns=self.valid_anndata_patterns,
            shuffle=self.shuffle,
            cycle=True,
        )

        valid_node, valid_cells = self.setup_node(
            self.valid_anndata_patterns,
            exclude_patterns=None,
            shuffle=False,
            cycle=False,
        )

        # Training: all_reduce MAX — every cell seen at least once, shorter
        # ranks cycle. Validation: all_reduce MIN — clean metrics, no re-eval.
        cells_tensor = torch.tensor(train_cells, dtype=torch.long)
        valid_cells_tensor = torch.tensor(valid_cells, dtype=torch.long)
        if torch.distributed.is_initialized():
            cells_tensor = cells_tensor.cuda()
            torch.distributed.all_reduce(cells_tensor, op=torch.distributed.ReduceOp.MAX)
            valid_cells_tensor = valid_cells_tensor.cuda()
            torch.distributed.all_reduce(valid_cells_tensor, op=torch.distributed.ReduceOp.MIN)
        self._train_batches = int(cells_tensor.item()) // self.batch_size
        self._valid_batches = int(valid_cells_tensor.item()) // self.batch_size

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.train_dataloader_node = train_node
        self.valid_dataloader_node = valid_node

    def train_dataloader(self):
        loader = Loader(self.train_dataloader_node)
        loader.root.reset()
        return _LenLoader(loader, self._train_batches)

    def val_dataloader(self):
        loader = Loader(self.valid_dataloader_node)
        loader.root.reset()
        return _LenLoader(loader, self._valid_batches)
