import sys
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torchdata.nodes import BaseNode

from celltransformer.loader.distributed import (
    get_gpu_rank_info,
    log_if_rank0,
    _get_cell_count,
    _round_robin_partition,
    maybe_shard_files_across_gpus,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AnnDataMetadataNode(BaseNode[str]):
    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = "",
        anndata_patterns: Optional[List[str]] = ["*.h5ad"],
        exclude_patterns: Optional[List[str]] = [],
        file_pair_list: Optional[List[str]] = None,
        shuffle_seed: Optional[int] = 1221,
        max_num_files: Optional[int] = None,
        is_child_node: bool = False,
        partition_strategy: str = "binpack",
        split_no: Optional[int] = None
    ):
        super().__init__()
        self.root_dir = Path(root_dir)

        if (
            file_pair_list is None
            and (not anndata_patterns)
            and (not exclude_patterns)
            and (not root_dir)
        ):
            raise ValueError("Either file_list or anndata_patterns must be provided.")

        if file_pair_list is not None and (
            anndata_patterns or exclude_patterns or root_dir
        ):
            raise ValueError(
                "file_list and anndata_patterns/exclude_patterns cannot be used together."
            )

        self.patterns = anndata_patterns
        self.exclude_patterns = exclude_patterns
        self._file_pairs = file_pair_list
        self._current_idx: int = 0
        self.seed = shuffle_seed
        self.max_num_files = max_num_files
        self.partition_strategy = partition_strategy
        self.rank, self.world_size, self.is_distributed = get_gpu_rank_info()
        self.split_no = split_no
        self.dist_log_str = f"[RANK: {self.rank}/{self.world_size} | DISTRIBUTED: {self.is_distributed} | SPLIT: {self.split_no}]"
        self.max_total_cells = None
        self.subsample_info = None  # (rank, world_size) when files are replicated across GPUs
        self._reset_count = 0

        if file_pair_list is not None:
            self._file_pairs = file_pair_list
            logger.info(f"{self.dist_log_str} - Child node initialized with {len(self._file_pairs)} file pairs (no additional sharding)")

        else:
            self._file_pairs = []
            for pattern in self.patterns:
                anndata_paths = sorted(list(self.root_dir.glob(pattern)))
                skip = False
                for anndata_path in anndata_paths:
                    # Skip if the file is in the exclude patterns
                    if self.exclude_patterns is not None:
                        for exclude_pattern in self.exclude_patterns:
                            if anndata_path.match(exclude_pattern):
                                log_if_rank0(
                                     f"Excluding file {anndata_path} due to exclude pattern {exclude_pattern}"
                                 )
                                skip = True

                    if skip:
                        skip = False
                        continue

                    metadata_path = self._get_metadata(anndata_path)
                    if metadata_path is not None:
                        # log_if_rank0(
                        #     f"Found metadata file {metadata_path} for {anndata_path}"
                        # )
                        self._file_pairs.append((anndata_path, metadata_path))
                    else:
                        log_if_rank0(f"No metadata file found for {anndata_path}; skipping!")
                        continue

            self._file_pairs = sorted(self._file_pairs)
            logger.info(f"{self.dist_log_str} - Found {len(self._file_pairs)} file pairs before sharding.")
            self._file_pairs, self.max_total_cells, self.subsample_info = maybe_shard_files_across_gpus(self._file_pairs, partition_strategy=self.partition_strategy)

        logger.info(f"{self.dist_log_str} - Found total of {len(self._file_pairs)} file pairs after sharding.")
            #logger.info(f"Using total of {len(self._file_pairs)} file pairs after sharding.")


    @classmethod
    def from_file_list(cls, file_list: List[Path], split_no: Optional[int] = None):
        if len(file_list) == 0:
            raise ValueError(f"Called AnnDataMetadataNode.from_file_list with empty file_list.")
        return cls(file_pair_list=file_list, anndata_patterns=[], is_child_node=True, split_no=split_no)

    def split(
        self, num_splits: int, as_dict: bool = True
    ) -> Union[List[tuple[str, str]], List[dict]]:
        logger.info(f"Splitting AnnDataMetadataNode into {num_splits} splits. Before split, file pairs: {self._file_pairs}")
        if self._file_pairs is None:
            self.reset()
            if len(self._file_pairs) == 0:
                raise ValueError(
                    "`.reset()` called but no files found. Input parsing patterns did not return anything."
                )

        logger.info(f"Splitting data into {num_splits} splits, from {len(self._file_pairs)} file pairs.")

        if num_splits < 1:
            raise ValueError("`num_splits` must be greater or equal to 1.")

        if num_splits > len(self._file_pairs):
            logger.info(
                f"`num_splits` is greater than the number of files."
            )
            num_splits = min(num_splits, len(self._file_pairs))

        # seed = self.seed
        # rank, _, is_distributed = get_gpu_rank_info()
        # rng = random.Random(seed)
        # rng.shuffle(self._file_pairs)

        if num_splits == 1:
            splits_map = [self._file_pairs]
        else:
            splits_map = _round_robin_partition(self._file_pairs, num_splits)

        nodes = []
        for idx, paths in enumerate(splits_map):
            nodes.append(AnnDataMetadataNode.from_file_list(paths, split_no=idx))

        if as_dict:
            return {f"s{i+1}": node for i, node in enumerate(nodes)}
        else:
            return nodes

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)
        if initial_state is not None:
            self._current_idx = initial_state["current_idx"]
            self._file_pairs = initial_state["file_pairs"]
            self.seed = initial_state["seed"]
            self._reset_count = initial_state.get("reset_count", self._reset_count)

        else:
            self._current_idx = 0
            self._reset_count += 1
            if self.seed is not None and len(self._file_pairs) > 1:
                rng = random.Random(self.seed + self._reset_count)
                rng.shuffle(self._file_pairs)

        rank, world_size, is_distributed = get_gpu_rank_info()
        if is_distributed:
            logger.info(f"Running on rank {rank} of {world_size} with {len(self._file_pairs)} files. {self._file_pairs}")

    def _get_metadata(self, file_path: Path) -> Union[None, Path]:
        maybe_csv_path = file_path.with_suffix(".csv")
        maybe_pq_path = file_path.with_suffix(".parquet")

        if maybe_csv_path.exists():
            return maybe_csv_path

        if maybe_pq_path.exists():
            return maybe_pq_path

        return None

    def next(self) -> tuple[str, str]:
        if self._current_idx >= len(self._file_pairs):
            raise StopIteration

        adata_path, metadata_path = self._file_pairs[self._current_idx]
        self._current_idx += 1
        return str(adata_path), str(metadata_path)

    def get_state(self) -> Dict[str, Any]:
        return { # TODO: handle saving RANK specific state if needed
            "current_idx": self._current_idx,
            "file_pairs": self._file_pairs,
            "seed": self.seed,
            "reset_count": self._reset_count,
        }
