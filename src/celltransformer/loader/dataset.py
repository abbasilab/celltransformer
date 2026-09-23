"""
TODO: https://github.com/pytorch/pytorch/issues/130825
Compile fails if used within a block that creates a NJT.
Apparently it may work if the code is refactored to use
formation of NJT outside of compiled sections.

"""

import sys
import pathlib
import logging
from typing import Dict, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy
import torch
from jaxtyping import Int
from scipy import sparse

from anndata._core.sparse_dataset import _CSRDataset as CSRDataset
#from anndata._core.sparse_dataset import CSRDataset

import polars as pl

from celltransformer.loader.types import NeighborhoodMetadata

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CenterMaskNode(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_path: str,
        adata_path: str,
        cell_type_colname: Optional[str],
        patch_size: Union[list[int, int], tuple[int, int]],
        tissue_section_colname: Optional[str] = None,
        use_gene_mask: bool = True,
        max_num_cells: Optional[int] = None,
        mode: Optional[str] = None,
    ):
        self.current_index = 0
        self.mode = mode
        self.max_num_cells = max_num_cells
        self.use_gene_mask = use_gene_mask
        self.tissue_section_colname = tissue_section_colname
        self.cell_type_colname = cell_type_colname
        self.patch_size = patch_size

        self.metadata = pathlib.Path(metadata_path)
        if self.metadata.suffix == '.csv':
            self.metadata = pl.read_csv(self.metadata)
        elif self.metadata.suffix == '.parquet':
            self.metadata = pl.read_parquet(self.metadata)

        self.metadata = self.metadata.with_row_index()

        self.adata = ad.read_h5ad(adata_path, backed=self.mode)


        self._preprocess_metadata()

        self.measured_gene_mask = None
        self.current_file_pair = None

    def __len__(self):
        return len(self.metadata)

    def _preprocess_metadata(self):
        if self.tissue_section_colname is None:
            return {}

        elif self.tissue_section_colname is not None:
            if self.tissue_section_colname not in self.metadata.columns:
                return {}
            else:
                tissue_section_groups = self.metadata.partition_by(
                    self.tissue_section_colname, as_dict=True
                )
                tissue_section_groups = {
                    key[0]: val for key, val in tissue_section_groups.items()
                }
                self.tissue_section_groups = tissue_section_groups


        if sparse.issparse(self.adata.X) or isinstance(self.adata.X, CSRDataset):
            self.idx_to_expr = self._idx_to_expr_sparse
        else:
            self.idx_to_expr = self._idx_to_expr_dense

        return tissue_section_groups

    def _idx_to_expr_sparse(self, idxs: Union[list[int], int]):
        return self.adata.X[idxs].toarray()

    def _idx_to_expr_dense(self, idxs: Union[list[int], int]):
        return self.adata.X[idxs]

    def idx_to_expr(self, idxs):
        pass

    def get_nearby_cells(self, x, y, exclude_idx, section: str = None) -> pl.DataFrame:
        xstart = x - self.patch_size[0] / 2.0
        xend = x + self.patch_size[0] / 2.0
        ystart = y - self.patch_size[1] / 2.0
        yend = y + self.patch_size[1] / 2.0

        if section:
            lookup = self.tissue_section_groups[section]
        else:
            lookup = self.metadata

        nearby_cells = (
            lookup.lazy()
            .filter(
                (pl.col("x") >= xstart)
                & (pl.col("x") <= xend)
                & (pl.col("y") >= ystart)
                & (pl.col("y") <= yend)
                & (pl.col("index") != exclude_idx)
            )
            .select(["index", self.cell_type_colname])
        ).collect()
        return nearby_cells

    def __getitem__(self, idx):
        # while True:
        #     if self.metadata_iter is None:
        #         # print("loading next dataset (maybe)")
        #         success, (metadata_path, adata_path) = self._load_next_dataset()
        #         self.current_file_pair = (metadata_path, adata_path)
        #         # print("success is ", success)
        #         if not success:
        #             raise StopIteration

        #     try:
        #         curr_row: dict = next(self.metadata_iter)
        curr_row = self.metadata.row(idx, named = True)
        nearby_cells_metadata = self.get_nearby_cells(
            curr_row["x"],
            curr_row["y"],
            exclude_idx=curr_row["index"],
            section=curr_row[self.tissue_section_colname]
            if self.tissue_section_colname
            else None,
        )

        if self.max_num_cells is not None:
            if len(nearby_cells_metadata) > self.max_num_cells:
                nearby_cells_metadata = nearby_cells_metadata.sample(
                    n=self.max_num_cells, shuffle=True
                )

        indices_to_get = [curr_row["index"]] + nearby_cells_metadata[
            "index"
        ].to_list()

        gex: np.ndarray = self.idx_to_expr(indices_to_get)
        hidden_gex = np.atleast_2d(gex[-1, :])  # 1 x g
        obs_gex = np.atleast_2d(gex[:-1, :])  # n x g

        if obs_gex.shape[0] != 0:
            return NeighborhoodMetadata(
                observed_expression=torch.from_numpy(obs_gex),
                masked_expression=torch.from_numpy(hidden_gex),
                masked_cell_type=torch.from_numpy(
                    np.array([curr_row[self.cell_type_colname]])
                ).long(),
                observed_cell_type=torch.from_numpy(
                    nearby_cells_metadata[self.cell_type_colname]
                    .to_numpy()
                    .copy()
                ).long(),
                num_cells_obs=len(obs_gex),
                measured_gene_mask=torch.from_numpy(
                    self.measured_gene_mask
                ).bool()
                if self.measured_gene_mask
                else None,
                section=curr_row.get(self.tissue_section_colname, None),
            )

        else:
            return NeighborhoodMetadata(
                observed_expression=torch.FloatTensor([]),
                masked_expression=torch.from_numpy(hidden_gex),
                masked_cell_type=torch.from_numpy(
                    np.array([curr_row[self.cell_type_colname]])
                ).long(),
                observed_cell_type=torch.FloatTensor([]),
                num_cells_obs=0,
                measured_gene_mask=torch.from_numpy(
                    self.measured_gene_mask
                ).bool()
                if self.measured_gene_mask
                else None,
                section=curr_row.get(self.tissue_section_colname, None),
            )

def index_outer_product(n: int) -> Int[np.ndarray, "seqlen 2"]:  # noqa: F722
    """Compute the product of the set of indices with itself and reshapes to (, 2). In other words,
    return all pairs of indices from the set (0, 1, ..., n).

    Returns
    -------
    array of int
        (n_indices, 2) np array.

    Examples
    --------
    >>> print(index_outer_product(3))
    array([[0, 0],
        [1, 0],
        [2, 0],
        [0, 1],
        [1, 1],
        [2, 1],
        [0, 2],
        [1, 2],
        [2, 2]])
    """
    i = np.arange(n)
    grid = np.dstack(np.meshgrid(i, i)).reshape(-1, 2)
    return grid


class CenterMaskNodeOptimized(torch.utils.data.Dataset):
    """
    Optimized version using simple filtering with performance improvements.
    Best for: sequential iteration through dataset, <100k cells per section
    """
    def __init__(
        self,
        metadata_path: str,
        adata_path: str,
        cell_type_colname: Optional[str],
        patch_size: Union[list[int, int], tuple[int, int]],
        tissue_section_colname: Optional[str] = None,
        x_col: str = "x",
        y_col: str = "y",
        use_gene_mask: bool = True,
        max_num_cells: Optional[int] = None,
        mode: str = "r",
        cell_type_int_map: Optional[Dict[str, int]] = None,
    ):
        self.current_index = 0
        self.mode = mode
        self.max_num_cells = max_num_cells
        self.use_gene_mask = use_gene_mask
        self.tissue_section_colname = tissue_section_colname
        self.cell_type_colname = cell_type_colname
        self.cell_type_int_map = cell_type_int_map
        self.patch_size = patch_size
        self.x_col = x_col
        self.y_col = y_col

        # Load metadata
        metadata_file = pathlib.Path(metadata_path)
        if metadata_file.suffix == '.csv':
            self.metadata = pd.read_csv(metadata_file)
        elif metadata_file.suffix == '.parquet':
            self.metadata = pd.read_parquet(metadata_file)

        # Ensure we have an index column
        self.metadata = self.metadata.reset_index(drop=True)
        #if 'index' not in self.metadata.columns:
        #    self.metadata = self.metadata.reset_index(drop=False)
        #else:
        #    self.metadata = self.metadata.reset_index(drop=True)

        # Load AnnData
        self.adata = ad.read_h5ad(adata_path)
        if scipy.sparse.issparse(self.adata.X):
            self.adata.X = self.adata.X.toarray()

        # Preprocess
        self._preprocess_metadata()

        self.measured_gene_mask = None
        self.current_file_pair = None

    def __len__(self):
        return len(self.metadata)

    def _preprocess_metadata(self):
        """Set up tissue section groups and expression getter."""
        # Apply label encoding if the cell-type column is not already integer.
        # cell_type_int_map is a {str_label: int} dict built once at the
        # training-script level from either a taxonomy YAML or LabelEncoder.
        if self.cell_type_int_map is not None:
            # .map() on a Categorical series can return a Categorical with int
            # categories rather than a plain numeric series; force int64 so
            # that .values later gives a plain numpy array, not a Categorical.
            self.metadata['_cell_type_int'] = (
                self.metadata[self.cell_type_colname]
                .map(self.cell_type_int_map)
                .astype(np.int64)
            )
            self._ct_col = '_cell_type_int'
        else:
            self._ct_col = self.cell_type_colname

        # Pre-compute section groups if needed
        if self.tissue_section_colname is not None and self.tissue_section_colname in self.metadata.columns:
            # Create a dictionary mapping section -> filtered dataframe
            self.tissue_section_groups = {
                name: group.copy() for name, group in self.metadata.groupby(self.tissue_section_colname, observed=True)
            }

            # For each section, sort cells by x-coordinate once at init time.
            # This lets __getitem__ narrow candidates to the x-band in O(log n)
            # via np.searchsorted, then filter y within that narrow slice.
            self.section_coords = {}
            for section, group in self.tissue_section_groups.items():
                x_vals = group[self.x_col].values
                sort_order = np.argsort(x_vals, kind='stable')
                self.section_coords[section] = {
                    'x': x_vals[sort_order],                        # sorted
                    'y': group[self.y_col].values[sort_order],
                    'index': group.index.values[sort_order],
                    'cell_type': group[self._ct_col].values[sort_order],
                }
        else:
            self.tissue_section_groups = {}
            # Sort entire dataframe by x for the same reason
            x_vals = self.metadata[self.x_col].values
            sort_order = np.argsort(x_vals, kind='stable')
            self.all_coords = {
                'x': x_vals[sort_order],
                'y': self.metadata[self.y_col].values[sort_order],
                'index': self.metadata['index'].values[sort_order],
                'cell_type': self.metadata[self._ct_col].values[sort_order],
            }

        # Pre-extract center-cell scalars into flat numpy arrays so __getitem__
        # never has to call metadata.iloc[idx] (which allocates a pandas Series).
        self._cx  = self.metadata[self.x_col].to_numpy()
        self._cy  = self.metadata[self.y_col].to_numpy()
        self._cct = self.metadata[self._ct_col].to_numpy()
        if self.tissue_section_colname and self.tissue_section_colname in self.metadata.columns:
            self._csec = self.metadata[self.tissue_section_colname].to_numpy()
        else:
            self._csec = None

        # Cache direct reference to the expression array to avoid the anndata
        # attribute chain on every __getitem__ call.
        if sparse.issparse(self.adata.X) or isinstance(self.adata.X, CSRDataset):
            self.idx_to_expr = self._idx_to_expr_sparse
        else:
            self._expr = self.adata.X
            self.idx_to_expr = self._idx_to_expr_dense

    def _idx_to_expr_sparse(self, idxs):
        return self.adata.X[idxs].toarray()

    def _idx_to_expr_dense(self, idxs):
        return self._expr[idxs]

    def get_nearby_cells_sorted(self, x, y, exclude_idx, section: str = None):
        """Find nearby cells using sorted-x + binary search (O(log n + k) per query).

        At init time coordinates are sorted by x, so searchsorted narrows the
        candidate set to the x-band before any element-wise work is done.
        The y filter and exclude mask are then applied only to that slice.
        """
        half_x = self.patch_size[0] / 2.0
        half_y = self.patch_size[1] / 2.0

        if section is not None and section in self.section_coords:
            coords = self.section_coords[section]
        else:
            coords = self.all_coords

        # Binary-search into the x-sorted array to get the x-band slice
        lo = np.searchsorted(coords['x'], x - half_x, side='left')
        hi = np.searchsorted(coords['x'], x + half_x, side='right')

        # Apply y filter and exclude-self mask only within the narrow slice
        y_slice    = coords['y'][lo:hi]
        idx_slice  = coords['index'][lo:hi]
        ct_slice   = coords['cell_type'][lo:hi]

        mask = (
            (y_slice >= y - half_y) &
            (y_slice <= y + half_y) &
            (idx_slice != exclude_idx)
        )

        return idx_slice[mask], ct_slice[mask]

    def __getitem__(self, idx):
        # Read center-cell scalars from pre-extracted numpy arrays (no pandas overhead)
        cx  = self._cx[idx]
        cy  = self._cy[idx]
        sec = self._csec[idx] if self._csec is not None else None

        # Use sorted-x binary-search filtering
        nearby_indices, nearby_cell_types = self.get_nearby_cells_sorted(
            cx, cy, exclude_idx=idx, section=sec,
        )

        # Sample if needed
        if self.max_num_cells is not None and len(nearby_indices) > self.max_num_cells:
            sample_mask = np.random.choice(len(nearby_indices), self.max_num_cells, replace=False)
            nearby_indices = nearby_indices[sample_mask]
            nearby_cell_types = nearby_cell_types[sample_mask]

        # Build expression index as a numpy array (avoids .tolist() + list concat)
        indices_to_get = np.empty(len(nearby_indices) + 1, dtype=np.intp)
        indices_to_get[0] = idx
        indices_to_get[1:] = nearby_indices

        # Get expression data
        gex: np.ndarray = self.idx_to_expr(indices_to_get)
        hidden_gex = np.atleast_2d(gex[0, :])  # First is the masked cell
        obs_gex = np.atleast_2d(gex[1:, :])    # Rest are observed

        cct = self._cct[idx]
        if obs_gex.shape[0] != 0:
            return NeighborhoodMetadata(
                observed_expression=torch.from_numpy(obs_gex),
                masked_expression=torch.from_numpy(hidden_gex),
                masked_cell_type=torch.tensor([cct], dtype=torch.long),
                observed_cell_type=torch.from_numpy(nearby_cell_types).long(),
                num_cells_obs=len(obs_gex),
                measured_gene_mask=torch.from_numpy(
                    self.measured_gene_mask
                ).bool()
                if self.measured_gene_mask is not None
                else None,
                section=sec,
            )
        else:
            return NeighborhoodMetadata(
                observed_expression=torch.FloatTensor([]),
                masked_expression=torch.from_numpy(hidden_gex),
                masked_cell_type=torch.tensor([cct], dtype=torch.long),
                observed_cell_type=torch.LongTensor([]),
                num_cells_obs=0,
                measured_gene_mask=torch.from_numpy(
                    self.measured_gene_mask
                ).bool()
                if self.measured_gene_mask is not None
                else None,
                section=sec,
            )
