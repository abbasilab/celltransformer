import heapq
import sys
import os
import logging
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def log_if_rank0(message: str):
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        logger.info(f"[Rank 0 ONLY]: {message}")

def get_gpu_rank_info() -> Tuple[int, int, bool]:
    """
    Detect if running in distributed mode and return (rank, world_size, is_distributed).

    Returns:
        tuple: (rank, world_size, is_distributed)
            - rank: GPU rank (0 if single GPU)
            - world_size: total number of GPUs (1 if single GPU)
            - is_distributed: True if running in multi-GPU mode
    """
    # Try torch.distributed first
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return (
            torch.distributed.get_rank(),
            torch.distributed.get_world_size(),
            True
        )

    # Fall back to environment variables (common in many distributed setups)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    return rank, world_size, is_distributed


def _get_cell_count(metadata_path) -> int:
    """Read cell count from a metadata file by reading only the parquet footer or CSV header."""
    metadata_path = Path(metadata_path)
    if metadata_path.suffix == '.parquet':
        import pyarrow.parquet as pq
        return pq.read_metadata(metadata_path).num_rows
    elif metadata_path.suffix == '.csv':
        with open(metadata_path) as f:
            return sum(1 for _ in f) - 1  # minus header
    else:
        raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")


def _round_robin_partition(file_pairs: List, world_size: int) -> List[List]:
    """Partition file_pairs across world_size groups using round-robin assignment.

    Returns:
        List of world_size lists, each containing file pairs for that rank.
    """
    groups = [[] for _ in range(world_size)]
    for i, fp in enumerate(file_pairs):
        groups[i % world_size].append(fp)
    return groups


def _greedy_binpack_partition(file_pairs: List, world_size: int) -> List[List]:
    """Partition file_pairs across world_size groups by greedy bin-packing.

    Files are sorted by descending cell count and assigned one at a time to the
    rank with the fewest total cells so far (min-heap). This minimises the
    spread between the heaviest and lightest rank, reducing wasted cells when
    training stops at the min-rank.

    Returns:
        List of world_size lists, each containing file pairs for that rank.
    """
    counted = [(pair, _get_cell_count(pair[1])) for pair in file_pairs]
    counted.sort(key=lambda x: x[1], reverse=True)

    # Min-heap of (total_cells, rank_index)
    heap = [(0, i) for i in range(world_size)]
    groups: List[List] = [[] for _ in range(world_size)]

    for pair, count in counted:
        total, rank_idx = heapq.heappop(heap)
        groups[rank_idx].append(pair)
        heapq.heappush(heap, (total + count, rank_idx))

    return groups


def maybe_shard_files_across_gpus(file_pairs: List, partition_strategy: Literal["binpack", "roundrobin"] = "binpack") -> Tuple[List, Optional[int], Optional[Tuple[int, int]]]:
    """Distribute file pairs across GPUs for balanced data loading.

    Three modes depending on the environment and number of files:

    1. Single GPU: All files go to the one GPU. No cell limit or subsampling.
       Returns: (all_file_pairs, None, None)

    2. Fewer files than GPUs: Every GPU gets ALL files, but each GPU only
       iterates a strided subset of cell indices within each file. For example,
       with 4 GPUs, rank 0 processes cells [0, 4, 8, ...], rank 1 processes
       cells [1, 5, 9, ...], etc. This avoids any GPU being idle.
       Returns: (all_file_pairs, None, (rank, world_size))

    3. Enough files for each GPU: Files are assigned to GPUs via round-robin
       so each GPU gets a disjoint set of files. A max_total_cells cap is set
       to the minimum group's cell count so all GPUs stop at the same point,
       preventing DDP hangs from uneven epoch lengths.
       Returns: (this_rank_file_pairs, max_total_cells, None)

    Returns:
        file_pairs: The file pairs this rank should load from.
        max_total_cells: If set, this rank should stop after yielding this many
            cells (ensures all ranks finish at the same time). None if not needed.
        subsample_info: If set, a (rank, world_size) tuple indicating this rank
            should only process every world_size-th cell starting at offset rank
            (stride-based index partitioning within shared files). None if each
            rank has its own exclusive files. Later we will use this to slice into
            cell indices provided to e.g. SubsetRandomSampler.
    """
    rank, world_size, is_distributed = get_gpu_rank_info()

    # Mode 1: single GPU — no sharding needed
    if not is_distributed:
        logger.info(f"Single GPU mode: using all {len(file_pairs)} files")
        return file_pairs, None, None

    # Mode 2: fewer files than GPUs — replicate files, partition cells by stride
    if len(file_pairs) < world_size:
        total_cells = sum(_get_cell_count(meta) for (_adata, meta) in file_pairs)
        cells_per_rank = total_cells // world_size
        logger.info(
            f"[GPU {rank}/{world_size}] Only {len(file_pairs)} files for {world_size} GPUs. "
            f"Replicating all files to every rank with stride-based subsampling. "
            f"Each rank gets ~{cells_per_rank} of {total_cells} cells."
        )
        return file_pairs, None, (rank, world_size)

    # Mode 3: enough files — partition files across GPUs, cap cells to min group
    if partition_strategy == "roundrobin":
        groups = _round_robin_partition(file_pairs, world_size)
        log_if_rank0(f"Using round-robin partition (partition_strategy={partition_strategy})")
    else:
        groups = _greedy_binpack_partition(file_pairs, world_size)
        log_if_rank0(f"Using greedy bin-pack partition (partition_strategy={partition_strategy})")

    group_cell_counts = []
    for g_idx, group in enumerate(groups):
        total = sum(_get_cell_count(meta) for (_adata, meta) in group)
        group_cell_counts.append(total)
        logger.info(f"[GPU {g_idx}/{world_size}] assigned {len(group)} files, {total} cells")

    min_cells = min(group_cell_counts)
    logger.info(
        f"[GPU {rank}/{world_size}] Cell counts per group: {group_cell_counts}. "
        f"Limiting all GPUs to {min_cells} cells."
    )

    return groups[rank], min_cells, None
