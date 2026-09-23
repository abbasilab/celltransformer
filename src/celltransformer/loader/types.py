from collections import namedtuple

NeighborhoodMetadata = namedtuple(
    "NeighborhoodMetadata",
    (
        "observed_expression",  # as float matrix, n_neighbor x g
        "masked_expression",  # as float matrix, n_masked (ex 1) x g
        "observed_cell_type",  # int vector of length n_neighbor
        "masked_cell_type",  # in vector of n_masked length
        "num_cells_obs",  # number of observed cells (cells in neighborhd)
        "measured_gene_mask",  # binary mask of genes that were measured
        "section",
    ),
)
