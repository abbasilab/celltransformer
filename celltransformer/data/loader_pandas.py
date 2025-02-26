from collections import namedtuple
from random import shuffle
from typing import Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from jaxtyping import Float, Int

NeighborhoodMetadata = namedtuple(
    "NeighborhoodMetadata",
    (
        "observed_expression",  # as float matrix, n_neighbor x g
        "masked_expression",  # as float matrix, n_masked (ex 1) x g
        "observed_cell_type",  # int vector of length n_neighbor
        "masked_cell_type",  # in vector of n_masked length
        "num_cells_obs",  # number of observed cells (cells in neighborhd)
    ),
)


def random_indices_from_series(df: Union[pd.Series, pd.DataFrame], n: int) -> List[int]:
    # randomly sample n indices from length of df/series
    indices = list(range(len(df)))
    shuffle(indices)
    return indices[:n]


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

def add_gaussian(xy, sigma):
    noise = np.random.standard_normal(size=xy.shape) * sigma
    return xy + noise

def collate(batched_metadata: NeighborhoodMetadata) -> Dict[str, torch.Tensor | int]:
    """Collates metadata which is a list of NeighborhoodMetadata namedtuples.
    One important computation is the attention matrices that the transformer will use.
    In particular the `encoder_mask` which is a square matrix with (n_obs_cells + n_cls_tokens [=bs]) length.
    The `pooling_mask` is a matrix of shape (n_pooling_tokens [=bs], n_obs_cells + n_cls_tokens) which is used to pool the hidden states.
    The `decoder_mask` is a square matrix of shape (n_pooled_tokens [=bs], n_query_tokens [=bs]) which is used to mask the decoding queries.
    The other items (the expression matrices and integer-encoded cell type vector) are simply concatenated.

    Parameters
    ----------
    batched_metadata : NeighborhoodMetadata
        Contains attributes: observed_expression, masked_expression, observed_cell_type, masked_cell_type, num_cells_obs
        Expression are float valued (n_cells x genes) matrices and cell types are integer-valued vectors.

    Returns
    -------
    dict
        A dictionary containing the collated metadata and the attention masks.
    """
    observed_expression = []
    masked_expression = []
    observed_cell_type = []
    masked_cell_type = []
    observed_neighboorhood_lens = []

    tot_num_obs_cells = sum([metadata.num_cells_obs for metadata in batched_metadata])
    bs = len(batched_metadata)

    encoder_mask = torch.ones(
        tot_num_obs_cells + bs * 2, tot_num_obs_cells + bs * 2, dtype=torch.bool
    )

    decoder_mask = torch.ones(bs * 2, bs * 2, dtype=torch.bool)
    pooling_mask = torch.ones(bs, bs + tot_num_obs_cells, dtype=torch.bool)

    offset = 0

    for i, metadata in enumerate(batched_metadata):
        num_hidden_cells = metadata.num_cells_obs
        pooling_tok_idx = (2 * bs) - i
        decoding_tok_idx = bs - i

        indices: Int[np.ndarray, "total_seqlen x_y"] = (  # noqa: F722
            index_outer_product(num_hidden_cells) + offset
        )

        pooling_indices: Int[np.ndarray, "tot_seqlen"] = np.full(  # noqa: F821
            len(indices), -pooling_tok_idx
        )

        # if more than one decoding (ie not just single masked cell)
        # need to form the product indices of (observed_cells, n_decoding_queries)
        # and (cls_tokens, decoding_queries)
        decoding_indices: Int[np.ndarray, "tot_seqlen"] = np.full(  # noqa: F821
            len(indices), -decoding_tok_idx
        )

        x = indices[:, 0]
        y = indices[:, 1]

        encoder_mask[x, y] = False
        encoder_mask[pooling_indices, y] = False
        encoder_mask[x, pooling_indices] = False
        encoder_mask[-pooling_tok_idx, -pooling_tok_idx] = False

        encoder_mask[decoding_indices, y] = False
        encoder_mask[x, decoding_indices] = False
        encoder_mask[decoding_indices, decoding_indices] = False

        encoder_mask[pooling_indices, decoding_indices] = False
        encoder_mask[decoding_indices, pooling_indices] = False

        pooling_mask[i, offset : offset + num_hidden_cells] = False

        decoder_mask[i, bs + i] = False
        decoder_mask[bs + i, i] = False

        offset += num_hidden_cells

        observed_expression.append(metadata.observed_expression)
        masked_expression.append(metadata.masked_expression)
        observed_cell_type.append(metadata.observed_cell_type)
        masked_cell_type.append(metadata.masked_cell_type)
        observed_neighboorhood_lens.append(metadata.num_cells_obs)

    observed_expression = torch.cat(observed_expression).float()
    masked_expression = torch.cat(masked_expression).float()
    observed_cell_type = torch.cat(observed_cell_type).long()
    masked_cell_type = torch.cat(masked_cell_type).long()

    num_cells_plus_cls = sum(observed_neighboorhood_lens) + bs

    return dict(
        observed_expression=observed_expression,
        masked_expression=masked_expression,
        observed_cell_type=observed_cell_type,
        masked_cell_type=masked_cell_type,
        observed_neighboorhood_lens=observed_neighboorhood_lens,
        full_mask=encoder_mask,
        pooling_mask=pooling_mask,
        encoder_mask=encoder_mask[:num_cells_plus_cls, :num_cells_plus_cls],
        decoder_mask=decoder_mask,
        bs=bs,
    )


class CenterMaskSampler(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        adata: ad.AnnData,
        patch_size: Union[List[int], Tuple[int]],
        cell_id_colname: Optional[str] = "cell_label",
        cell_type_colname: Optional[str] = "cell_type",
        tissue_section_colname: Optional[str] = "brain_section_label",
        max_num_cells: Optional[Union[int, None]] = None,
        indices: Optional[Union[List[int], None]] = None,
    ):
        """Sampler that returns the gene expression matrices and cell-type identity vectors
         for two groups of cells: the observed cells and the masked/reference cells. The observed cells
         are the cells in the neighborhood of the reference cell (set by `patch_size`).

         We assume the type will be found at `cell_type_colname` and that `cell_id` can be used
         succesfully to index into the adata object.

        Parameters
        ----------
        metadata : pd.DataFrame
            Metadata for the cells. Must contain columns for cell_id, cell_type, x, y, and tissue_section.
        adata : ad.AnnData
            Expression-containing (as .X) anndata. We assume input will be log scaled.
        patch_size : Union[List[int], Tuple[int]]
            Size in arbitrary units for the neighborhood calculation.
        cell_id_colname : Optional[str], optional
            The column to use to index into the anndata, by default "cell_label"
        cell_type_colname : Optional[str], optional
            The column to use to access the cell type identities as cls-encoding integers, by default "cell_type"
        tissue_section_colname : Optional[str], optional
            To simplify computation, group the cells in each sectionsby this column, by default "brain_section_label"
        max_num_cells : Optional[Union[int, None]], optional
            How many cells to threshold at for the neighborhood size, by default None
        indices: Optional[Union[List[int], None]], optional
            Used to specify train/test sets via subsetting on only these cells. This should be a numeric index compatible with 
            `.iloc`, so it may be advisable to reset the index of the dataframe. By default None

        Raises
        ------
        TypeError
            If adata is not an `ad.annData` object
        ValueError
            If metadata does not contain the necessary columns
        TypeError
            If metadata is not a pandas DataFrame
        ValueError
            If patch_size is not a tuple of length 2
        ValueError
            If metadata and adata are not the same length
        ValueError
            If metadata does not contain columns "x" and "y"
        """
        if isinstance(adata, ad.AnnData) is False:
            raise TypeError(
                f"Input argument adata must be of type np.ndarray; got {type(adata)}."
            )

        if not all(
            (
                cell_type_colname in metadata.columns,
                cell_id_colname in metadata.columns,
                tissue_section_colname in metadata.columns,
            )
        ):
            raise ValueError(
                "Provided metadata has to have columns with columns specified by cell_id_colname, cell_type_colname and tissue_section_colname."
            )

        if isinstance(metadata, pd.DataFrame) is False:
            raise TypeError(
                f"Input argument metadata must be of type pl.DataFrame or pd.DataFrame; got {type(metadata)}."
            )

        if len(patch_size) != 2:
            raise ValueError(
                f"Input argument patch_size must be a tuple of length 2 with desired dims order (x, y, z); got {len(patch_size)}."
            )

        if len(metadata) != len(adata):
            raise ValueError("Metadata and adata must have the same length.")

        if not ("x" in metadata.columns and "y" in metadata.columns):
            raise ValueError(
                "Metadata must contain columns named x and y to index the spatial coordinates of the cells."
            )

        self.metadata = metadata

        # use these later to index into metadata dataframe
        self.cell_type_colname = cell_type_colname
        self.cell_id_colname = cell_id_colname
        self.tissue_section_colname = tissue_section_colname
        self.length = len(self.metadata) if indices is None else len(indices)
        self.max_num_cells = np.inf if max_num_cells is None else max_num_cells

        self.adata = adata
        self.patch_size = np.array(patch_size).astype(int)
        #self.noise_fac = 2

        self._preprocess_metadata()

        # so that later we can simply pass a train/valid/test set of indices
        # instead of filtering up front
        self.indices = indices if indices is not None else list(range(self.length))

    def __len__(self):
        return self.length

    def _preprocess_metadata(self):
        # for use in reducing complexity of neighborhood queries in getitem
        self.tissue_section_mapper = {
            section_label: subset_df
            for section_label, subset_df in self.metadata.groupby(
                self.tissue_section_colname
            )
        }

    def get_nearby_cells(
        self,
        centroid: npt.ArrayLike,
        patch_size: npt.ArrayLike,
        tissue_section: pd.DataFrame,
        discretize: bool = False,
    ) -> pd.DataFrame:
        """Return the cells from tissue_section that are within
        patch_size distance away from centroid.

        Parameters
        ----------
        centroid : npt.ArrayLike
            A 1x2 vector representing the centroid of the patch.
        patch_size : npt.ArrayLike
            A 1x2 vector representing the size of the patch.
        tissue_section : pd.DataFrame
            A pandas DataFrame containing the cells in the tissue section.
            It's assumed that the cell corresponding to centroid is in this dataframe.
        discretize : bool, optional
            _description_, by default False

        Returns
        -------
        pd.DataFrame
            Cells inside the neighborhood, inclusive of the
            centroid cell / reference cell.
        """
        start = centroid - (patch_size / 2)
        if discretize:
            start = np.rint(start).astype(int)

        end = start + patch_size

        lookup_x = tissue_section["x"].between(start[0], end[0])
        lookup_y = tissue_section["y"].between(start[1], end[1])

        return tissue_section[lookup_x & lookup_y]

    def idx_to_expr(self, indices: Union[str, int, List[Union[str, int]]]):
        return self.adata[indices].X

    def __getitem__(self, index) -> dict:
        orig_idx = self.indices[index]  # see init for expln
        metadata_row = self.metadata.iloc[orig_idx]
        index = metadata_row[self.cell_id_colname]

        centroid = np.array([metadata_row["x"], metadata_row["y"]])
#        centroid = add_gaussian(centroid, self.noise_fac)

        tissue_section: pd.DataFrame = self.tissue_section_mapper.get(
            metadata_row[self.tissue_section_colname], None
        )

        assert tissue_section is not None, (
            "Tissue section not found in `tissue_section_mapper` dict. "
            "Check that the tissue_section_colname is correctly set in the metadata."
        )

        all_neighborhood_cells = self.get_nearby_cells(
            centroid, self.patch_size, tissue_section, discretize=True
        )

        if len(all_neighborhood_cells) == 1:
            # manually create an empty neighborhood
            masked_expression: Float[np.array, "one n_genes"] = self.idx_to_expr(index)  # noqa: F722
            # check if is csr, if so convert
            if not isinstance(masked_expression, np.ndarray):
                masked_expression = masked_expression.toarray()
            masked_cell_type: int = metadata_row[self.cell_type_colname]

            neighborhood_metadata = NeighborhoodMetadata(
                observed_expression=torch.FloatTensor([]),
                masked_expression=torch.from_numpy(masked_expression),
                observed_cell_type=torch.LongTensor([]),
                masked_cell_type=torch.LongTensor([masked_cell_type]),
                num_cells_obs=0,
            )
            return neighborhood_metadata

        return self.neighborhood_gather(all_neighborhood_cells, index, as_dict=False)

    def neighborhood_gather(
        self,
        neighborhood_cells: pd.DataFrame,
        ref_cell_label: str,
        as_dict: Optional[bool] = False,
    ):
        """Get the neighborhood cells' metadata and partition them into two sets.
        In this case we will only predict the center / reference cell.

        Parameters
        ----------
        neighborhood_cells : pd.DataFrame
            Dataframe containing the cells in the neighborhood. One
            cell's metadata per row.
        ref_cell_label : str
            The label of the reference cell as a string.
        as_dict : Optional[bool], optional
            Return a dictionary (True) or NeighborhoodMetadata (False),
            by default False.

        Returns
        -------
        Union[NeighborhoodMetadata, dict]
            A namedtuple or dictionary containing the partitioned cells'
            metadata, meaning the expression and class label integer for
            each of the cells. The number of observed cells is also returned,
            so in total there will be five entries / keys.

        """

        neighborhood_cells = neighborhood_cells.reset_index(drop=True)
        neighborhood_cells_indices = neighborhood_cells[self.cell_id_colname]
        expression = self.adata[neighborhood_cells_indices].X
        
        # check if is csr, if so convert
        if not isinstance(expression, np.ndarray):
            expression = expression.toarray()

        num_cells_neighborhood = len(neighborhood_cells)

        observed_cells: pd.DataFrame = neighborhood_cells[
            neighborhood_cells_indices != ref_cell_label
        ]

        masked_cell = neighborhood_cells[neighborhood_cells_indices == ref_cell_label]

        if self.max_num_cells < (num_cells_neighborhood - 1):
            random_indices: List[int] = random_indices_from_series(
                observed_cells[self.cell_id_colname], self.max_num_cells
            )
            observed_cells = observed_cells.iloc[random_indices]

        observed_cell_types = observed_cells[self.cell_type_colname].values
        observed_expression: Float[np.ndarray, "cells genes"] = expression[observed_cells.index] # noqa: F722

        masked_cell_types = masked_cell[self.cell_type_colname].values
        masked_expression: Float[np.ndarray, "cells genes"] = expression[masked_cell.index] # noqa: F722

        num_cells_obs = len(observed_cells)

        if as_dict:
            return dict(
                observed_expression=observed_expression,
                masked_expression=masked_expression,
                observed_cell_type=observed_cell_types,
                masked_cell_type=masked_cell_types,
                num_cells_obs=num_cells_obs,
            )
        else:
            return NeighborhoodMetadata(
                torch.from_numpy(observed_expression),
                torch.from_numpy(masked_expression),
                torch.from_numpy(observed_cell_types),
                torch.from_numpy(masked_cell_types),
                num_cells_obs,
            )
