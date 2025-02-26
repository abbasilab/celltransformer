from collections import namedtuple
from random import shuffle
from typing import List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import torch
from jaxtyping import Float, Int

"""
For some reason I couldn't make polars faster than pandas; as it stands there is a 
substantial speedup from pandas -- in addition, it appears that the multiprocessing 
that polars does interferes somehow with the torch DataLoader. 

I left this in as a reference in case others have helpful ideas on how to speed up the
polars implementation.
"""

NeighborhoodMetadata = namedtuple(
    "NeighborhoodMetadata",
    (  
        "observed_expression",
        "masked_expression",
        "observed_cell_type",
        "masked_cell_type",
        "num_cells_obs",
    ),
)


def random_indices_from_series(df: pl.Series, n: int) -> List[int]:
    indices = list(range(len(df)))
    shuffle(indices)
    return indices[:n]


def index_outer_product(n: int) -> Int[np.ndarray, "seqlen x_y_ind"]:  # noqa: F722
    i = np.arange(n)
    grid = np.meshgrid(i, i).reshape(-1, 2)
    return grid


def collate(batched_metadata: NeighborhoodMetadata):
    observed_expression = []
    masked_expression = []
    observed_cell_type = []
    masked_cell_type = []
    observed_neighboorhood_lens = []

    tot_num_obs_cells = sum([metadata.num_cells_obs for metadata in batched_metadata])
    bs = len(batched_metadata)

    attn_mask = torch.ones(
        tot_num_obs_cells + bs * 2, tot_num_obs_cells + bs * 2, dtype=torch.bool
    )

    decoder_mask = torch.ones(bs * 2, bs * 2, dtype=torch.bool)

    offset = 0

    for i, metadata in enumerate(batched_metadata):
        num_hidden_cells = metadata.num_cell_obs
        pooling_tok_idx = bs - i

        pooling_indices: Int[np.ndarray, "total_seqlen"] = np.full(  # noqa: F821
            num_hidden_cells, pooling_tok_idx
        )
        indices: Int[np.ndarray, "total_seqlen x_y"] = (  # noqa: F722
            index_outer_product(num_hidden_cells) + offset
        )

        x = indices[:, 0]
        y = indices[:, 1]

        attn_mask[x, y] = False
        attn_mask[pooling_indices, y] = False
        attn_mask[x, pooling_indices] = False
        attn_mask[pooling_tok_idx, pooling_tok_idx] = False

        decoder_mask[i, bs + i] = False
        decoder_mask[bs + i, i] = False

        offset += num_hidden_cells

        observed_expression.expression.append(metadata.observed_expression)
        masked_expression.expression.append(metadata.masked_expression)
        observed_cell_type.append(metadata.observed_cell_type)
        masked_cell_type.append(metadata.masked_cell_type)
        observed_neighboorhood_lens.append(metadata.num_cells_obs)

    observed_expression = torch.cat(observed_expression)
    masked_expression = torch.cat(masked_expression)
    observed_cell_type = torch.cat(observed_cell_type).long()
    masked_cell_type = torch.cat(masked_cell_type).long()

    return dict(
        observed_expression=observed_expression,
        masked_expression=masked_expression,
        observed_cell_type=observed_cell_type,
        masked_cell_type=masked_cell_type,
        observed_neighboorhood_lens=observed_neighboorhood_lens,
        attn_mask=attn_mask,
        decoder_mask=decoder_mask,
        bs=bs,
    )


class CenterMaskSampler(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata: Union[pd.DataFrame, pl.DataFrame],
        adata: ad.AnnData,
        patch_size: Union[List[int], Tuple[int]],
        cell_id_colname: Optional[str] = "cell_label",
        cell_type_colname: Optional[str] = "cell_type",
        tissue_section_colname: Optional[str] = "brain_section_label",
        max_num_cells: Optional[Union[int, None]] = None
    ):
        if isinstance(adata, ad.AnnData) is False:
            raise TypeError(
                f"Input argument adata must be of type np.ndarray; got {type(adata)}."
            )

        if not all((
            cell_type_colname in metadata.columns,
            cell_id_colname in metadata.columns,
            tissue_section_colname in metadata.columns)
        ):
            raise ValueError(
                "Provided metadata has to have columns with columns specified by cell_id_colname, cell_type_colname and tissue_section_colname."
            )

        if isinstance(metadata, (pd.DataFrame, pl.DataFrame)) is False:
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

        self.metadata = (
            metadata
            if isinstance(metadata, pl.DataFrame)
            else pl.from_dataframe(metadata)
        )
        self.cell_type_colname = cell_type_colname
        self.cell_id_colname = cell_id_colname
        self.tissue_section_colname = tissue_section_colname
        self.length = len(self.metadata)
        self.max_num_cells = np.inf if max_num_cells is None else max_num_cells

        self.adata = adata
        self.patch_size = np.array(patch_size).astype(int)

        self._preprocess_metadata()

    def __len__(self):
        return self.length

    def _preprocess_metadata(self):
        self.tissue_section_mapper = {
            section_label: subset_df
            for section_label, subset_df in self.metadata.group_by(
                self.tissue_section_colname
            )
        }

    def get_nearby_cells(
        self,
        centroid: npt.ArrayLike,
        patch_size: npt.ArrayLike,
        tissue_section: pd.DataFrame,
        discretize: bool = False,
    ) -> pl.DataFrame:
        
        start = centroid - (patch_size / 2)
        if discretize:
            start = np.rint(start).astype(int)

        end = start + patch_size

        lookup_x = pl.col('x').is_between(start[0], end[0])
        lookup_y = pl.col('y').is_between(start[1], end[1])

        return tissue_section.filter(lookup_x & lookup_y)

    def idx_to_expr(self, indices: Union[str, int, List[Union[str, int]]]):
        return self.adata[indices].X

    def __getitem__(self, index) -> dict:
        metadata_row: dict = self.metadata.row(index, named=True)
        index = metadata_row[self.cell_id_colname]

        centroid = np.array([metadata_row["x"], metadata_row["y"]])
        tissue_section: Union[pd.DataFrame, None] = self.tissue_section_mapper.get(
            metadata_row[self.tissue_section_colname], None
        )

        if tissue_section is None:
            raise RuntimeError(
                "Tissue section not found in `tissue_section_mapper` dict."
            )

        all_neighborhood_cells = self.get_nearby_cells(
            centroid, self.patch_size, tissue_section, discretize=False
        )

        if len(all_neighborhood_cells) == 1:
            masked_expression: Float[np.array, "one n_genes"] = self.idx_to_expr(index)  # noqa: F722
            masked_cell_type: int = metadata_row[self.cell_type_colname]

            neighborhood_metadata = NeighborhoodMetadata(
                observed_expression=torch.FloatTensor([]),
                masked_expression=torch.from_numpy(masked_expression),
                observed_cell_type=torch.LongTensor([]),
                masked_cell_type=torch.from_numpy([masked_cell_type]).long(),
                num_cells=1,
            )
            return neighborhood_metadata

        return self.neighborhood_gather(all_neighborhood_cells, index)

    def neighborhood_gather(
        self, neighborhood_cells: pl.DataFrame, ref_cell_label: str, as_dict: Optional[bool] = False
    ):
        neighborhood_cells_indices = neighborhood_cells[self.cell_type_colname]
        #expression = self.adata[neighborhood_cells_indices].X

        num_cells_neighborhood = len(neighborhood_cells)

        observed_cells: pl.DataFrame = neighborhood_cells.filter(
            neighborhood_cells_indices != ref_cell_label
        )
        masked_cell = neighborhood_cells.filter(pl.col(self.cell_id_colname) == ref_cell_label)
        observed_cells = neighborhood_cells.filter(pl.col(self.cell_id_colname) != ref_cell_label)

        if self.max_num_cells < num_cells_neighborhood - 1:
            random_indices: List[int] = random_indices_from_series(
                observed_cells[self.cell_id_colname], self.max_num_cells
            )
            observed_cells = observed_cells.with_row_index().filter(
                pl.col("index").is_in(random_indices)
            )

        observed_cell_types = observed_cells[self.cell_type_colname]
        observed_expression: npt.ArrayLike = self.idx_to_expr(
            observed_cells[self.cell_id_colname].to_list()
        )

        masked_cell_types = masked_cell[self.cell_type_colname]
        masked_expression: npt.ArrayLike = self.idx_to_expr(
            masked_cell[self.cell_id_colname].to_list()
        )

        num_cells_obs = len(observed_cells)

        if as_dict:
            return dict(
                observed_expression=observed_expression,
                masked_expression=masked_expression,
                observed_cell_type=observed_cell_types,
                masked_cell_type=masked_cell_types,
            )
        else:
            return NeighborhoodMetadata(
                torch.from_numpy(observed_expression),
                torch.from_numpy(masked_expression),
                torch.from_numpy(observed_cell_types).long(),
                torch.from_numpy(masked_cell_types).long(),
                num_cells_obs,
            )
