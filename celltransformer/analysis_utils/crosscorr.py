import torch
from jaxtyping import Float


def normalize_matrix(a):
    # zero mean the vectors
    return a - a.mean(1)[:, None]


def corr_predictions(
    a: Float[torch.Tensor, "ncells ngenes"], b: Float[torch.Tensor, "ncells ngenes"]    # noqa: F722
) -> Float[torch.Tensor, "ncells"]:    # noqa: F821
    """
    Computes the cross correlation between two pairs of matrices of the same length.
    Instantiating the entire matrix is unnecessary, so we just compute the pairwise 
    distances between the (i,i) entries in the two matrices.

    Parameters
    ---------
    a: Float[torch.Tensor, "ncells ngenes"] 
        obs by features matrix
    b: Float[torch.Tensor, "ncells ngenes"] 
        obs by features matrix; checked to be same size as `a`

    Returns
    -------
    Float[torch.Tensor, "ncells"] 
        Cross correlations where each element is Pearson(a_i, b_i) for i in num_obs.
    """

    assert a.shape == b.shape, "a and b must have the same shape"
    n_el = a.shape[1]

    a = normalize_matrix(a)
    b = normalize_matrix(b)

    sigma_a = a.std(1)
    sigma_b = b.std(1)

    # ommitting .item() will cause a shape error if args are torch.Tensor
    # otherwise this code will also work for numpy arrays
    if sigma_a.size == 1:
        sigma_a = sigma_a.item()

    if sigma_b.size == 1:
        sigma_b = sigma_b.item()

    prod = sigma_a * sigma_b

    cov = (a * b).sum(1) / n_el
    return cov / prod
