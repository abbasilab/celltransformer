import sys
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torchdata.nodes import (
    BaseNode,
    ParallelMapper,
    SamplerWrapper,
)
from torch.utils.data import RandomSampler, SequentialSampler

from celltransformer.loader.types import NeighborhoodMetadata
from celltransformer.loader.dataset import CenterMaskNodeOptimized

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CellMapper:
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

class CellParallelSampler(BaseNode):
    def __init__(self,
                 source: BaseNode,
                 n_workers: int,
                 cell_type_colname: Optional[str],
                 patch_size: Union[list[int, int], tuple[int, int]],
                 tissue_section_colname: Optional[str] = None,
                 use_gene_mask: bool = True,
                 max_num_cells: Optional[int] = None,
                 max_total_cells: Optional[int] = None,
                 subsample_info: Optional[Tuple[int, int]] = None,
                 shuffle: Optional[bool] = False,
                 method: Optional[str] = 'thread',
                 read_mode: Optional[str] = None,
                 bs: Optional[bool] = 128,
                 dist_log_str: Optional[str] = None,
                 cell_type_int_map: Optional[Dict[str, int]] = None,
                 use_cell_type: bool = True):

        self.source = source
        self.n_workers = n_workers
        self.current_sampling_node = None
        self.cell_type_colname = cell_type_colname
        self.cell_type_int_map = cell_type_int_map
        self.use_cell_type = use_cell_type
        self.patch_size = patch_size
        self.tissue_section_colname = tissue_section_colname
        self.max_num_cells = max_num_cells
        self.max_total_cells = max_total_cells
        self.subsample_info = subsample_info  # (rank, world_size) for stride-based index partitioning
        self._cells_yielded = 0
        self.use_gene_mask = use_gene_mask
        self.current_file_pair = None
        self.current_dataset = None
        self.shuffle = shuffle
        self.mode = read_mode
        self.bs = bs
        self.current_node = None
        self.method = method
        self.dist_log_str = dist_log_str
        self._reset_count = 0
        self._file_load_count = 0

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        self._cells_yielded = 0
        self.current_node = None
        self._reset_count += 1
        self._file_load_count = 0
        self.source.reset()

    def _load_next_file(self):
        try:
            adata_path, metadata_path = next(self.source)
            logger.info(f"{self.dist_log_str} - Loading new file pair: {metadata_path}") #, {adata_path}")
            #else:
            #    logger.info(f"{self.dist_log_str} - Loading new file pair: {metadata_path}, {adata_path}")

            return True, (metadata_path, adata_path)

        except StopIteration:
            return False, ("", "")

    def next(self):
        while True:
            # Stop if we've hit the cell limit for this GPU
            if self.max_total_cells is not None and self._cells_yielded >= self.max_total_cells:
                logger.info(
                    f"{self.dist_log_str} - Reached max_total_cells limit "
                    f"({self._cells_yielded}/{self.max_total_cells}). Stopping."
                )
                raise StopIteration

            if self.current_node is None:
                success, (metadata_path, adata_path) = self._load_next_file()

                if not success:
                    raise StopIteration

                self.current_file_pair = (metadata_path, adata_path)
                #self.current_dataset = CenterMaskNode(
                self.current_dataset =  CenterMaskNodeOptimized(
                    metadata_path, adata_path,
                    cell_type_colname = self.cell_type_colname,
                    patch_size = self.patch_size,
                    tissue_section_colname = self.tissue_section_colname,
                    max_num_cells = self.max_num_cells,
                    mode = self.mode,
                    cell_type_int_map = self.cell_type_int_map,
                    use_cell_type = self.use_cell_type,
                )

                if self.subsample_info is not None:
                    # Stride-based partitioning: each rank gets disjoint cell indices
                    sub_rank, sub_world = self.subsample_info
                    rank_indices = list(range(sub_rank, len(self.current_dataset), sub_world))
                    sampler = torch.utils.data.SubsetRandomSampler(rank_indices)
                else:
                    if self.shuffle:
                        g = torch.Generator()
                        g.manual_seed(1221 + self._reset_count * 100003 + self._file_load_count)
                        sampler = RandomSampler(self.current_dataset, generator=g)
                    else:
                        sampler = SequentialSampler(self.current_dataset)
                self._file_load_count += 1
                node = SamplerWrapper(sampler)
                #node = Batcher(node, batch_size=self.bs)
                node = ParallelMapper(node, self.current_dataset.__getitem__,
                                        num_workers=self.n_workers,
                                        method=self.method
                                      )
                node.reset()
                self.current_node = node
            try:
                next_data = self.current_node.next()
                self._cells_yielded += 1
                return next_data
            except StopIteration:
                self.current_node = None
                continue

    def get_state(self):
        return {}


class BudgetBatcher(BaseNode):
    """Batch NeighborhoodMetadata items by count AND total-cell token budget.

    Each item costs ``item.num_cells_obs + 1`` tokens (neighbors + masked cell).
    Items with ``num_cells_obs == 0`` are silently skipped — no-neighborhood
    cells cannot be used by the model and would waste batch slots.

    When adding the next item would push the running token total past
    ``max_tokens``, the item is held in a one-item ``_pending`` buffer and
    becomes the first item of the next batch.  This ensures no batch ever
    exceeds the budget while also guaranteeing every item is eventually
    returned (nothing is dropped due to the budget).

    ``next()`` always either returns a non-empty list or raises StopIteration —
    it never returns None.
    """

    def __init__(
        self,
        source: BaseNode,
        max_batch_size: int,
        max_tokens: Optional[int] = None,
    ):
        super().__init__()
        self.source = source
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens
        self._pending: Optional[NeighborhoodMetadata] = None

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        self._pending = None
        self.source.reset()

    def get_state(self) -> Dict[str, Any]:
        return {}

    def _next_valid(self) -> Optional[NeighborhoodMetadata]:
        """Pull from source, skipping zero-neighbor items.

        Returns the next item with num_cells_obs > 0, or None when the
        source is exhausted.  The caller (next()) decides what to do with None.
        """
        while True:
            try:
                item = self.source.next()
            except StopIteration:
                return None
            if item.num_cells_obs > 0:
                return item
            # num_cells_obs == 0: silently skip, loop to try next item

    def next(self) -> List[NeighborhoodMetadata]:
        batch: List[NeighborhoodMetadata] = []
        tokens: int = 0

        # Carry over any item that overflowed the previous batch budget
        if self._pending is not None:
            item = self._pending
            self._pending = None
            batch.append(item)
            tokens = item.num_cells_obs + 1

        while (len(batch) < self.max_batch_size) and (self.max_tokens is None or tokens < self.max_tokens):
            item = self._next_valid()
            if item is None:
                break  # source exhausted; return whatever we have

            item_tokens = item.num_cells_obs + 1

            if (
                self.max_tokens is not None
                and batch  # never split on the very first item in the batch
                and tokens + item_tokens > self.max_tokens
            ):
                self._pending = item  # hold for next batch
                break

            batch.append(item)
            tokens += item_tokens

        if not batch:
            raise StopIteration

        return batch
