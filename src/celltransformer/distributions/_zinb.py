"""Negative Binomial and Zero-Inflated Negative Binomial distributions.

Vendored from scvi-tools (via brainformr) to remove the scvi dependency.
Only the (mu, theta) parameterization and log_prob() are needed for training.

Reference:
    Lopez et al., "Deep generative modeling for single-cell transcriptomics",
    Nature Methods, 2018. https://doi.org/10.1038/s41592-018-0229-2
"""

import warnings

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all, probs_to_logits

from ._constraints import optional_constraint


def log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Log likelihood of a minibatch under a negative binomial model.

    Parameters
    ----------
    x
        Observed counts.
    mu
        Mean of the NB (positive support).
    theta
        Inverse dispersion (positive support).
    eps
        Numerical stability constant.
    """
    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    return res


def log_zinb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Log likelihood of a minibatch under a zero-inflated negative binomial.

    Parameters
    ----------
    x
        Observed counts.
    mu
        Mean of the NB component (positive support).
    theta
        Inverse dispersion (positive support).
    pi
        Logit of the dropout parameter (real support).
    eps
        Numerical stability constant.
    """
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    return mul_case_zero + mul_case_non_zero


class NegativeBinomial(Distribution):
    r"""Negative binomial distribution.

    Supports two parameterizations:

    1. (``total_count``, ``probs``/``logits``) — standard PyTorch convention.
    2. (``mu``, ``theta``) — mean and inverse dispersion, used by scvi-tools.

    In the (mu, theta) parameterization, samples are generated as:

    .. math::
        w \sim \text{Gamma}(\theta, \theta / \mu), \quad x \sim \text{Poisson}(w)

    Parameters
    ----------
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    scale
        Normalized mean expression (stored, not used in computations).
    """

    arg_constraints = {
        "mu": optional_constraint(constraints.greater_than_eq(0)),
        "theta": optional_constraint(constraints.greater_than_eq(0)),
        "scale": optional_constraint(constraints.greater_than_eq(0)),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        mu: torch.Tensor | None = None,
        theta: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        validate_args: bool = False,
    ):
        self._eps = 1e-8
        if (mu is None) == (total_count is None):
            raise ValueError(
                "Exactly one of (mu, theta) or (total_count, probs/logits) "
                "must be provided."
            )

        using_param_1 = total_count is not None and (
            logits is not None or probs is not None
        )
        if using_param_1:
            logits = logits if logits is not None else probs_to_logits(probs)
            total_count = total_count.type_as(logits)
            total_count, logits = broadcast_all(total_count, logits)
            mu = logits.exp() * total_count
            theta = total_count
            scale = mu / torch.sum(mu, dim=-1, keepdim=True)
        else:
            mu, theta = broadcast_all(mu, theta)

        self.mu = mu
        self.theta = theta
        self.scale = scale
        super().__init__(validate_args=validate_args)

    def __repr__(self) -> str:
        param_names = [k for k, _ in self.arg_constraints.items() if k in self.__dict__]
        args_string = ", ".join(
            f"{p}: {self.__dict__[p].size()}"
            for p in param_names
            if self.__dict__[p] is not None
        )
        return f"{self.__class__.__name__}({args_string})"

    @property
    def mean(self) -> torch.Tensor:
        return self.mu

    @property
    def variance(self) -> torch.Tensor:
        return self.mean + (self.mean**2) / self.theta

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            try:
                self._validate_sample(value)
            except ValueError:
                warnings.warn(
                    "The value argument must be within the support of the distribution",
                    UserWarning,
                    stacklevel=2,
                )
        return log_nb_positive(value, mu=self.mu, theta=self.theta, eps=self._eps)


class ZeroInflatedNegativeBinomial(NegativeBinomial):
    r"""Zero-inflated negative binomial distribution.

    Extends :class:`NegativeBinomial` with a Bernoulli zero-inflation component.

    Parameters
    ----------
    mu
        Mean of the NB component.
    theta
        Inverse dispersion.
    zi_logits
        Logits of the zero-inflation probability.
    scale
        Normalized mean expression (stored, not used in computations).
    """

    arg_constraints = {
        "mu": optional_constraint(constraints.greater_than_eq(0)),
        "theta": optional_constraint(constraints.greater_than_eq(0)),
        "zi_logits": optional_constraint(constraints.real),
        "scale": optional_constraint(constraints.greater_than_eq(0)),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        mu: torch.Tensor | None = None,
        theta: torch.Tensor | None = None,
        zi_logits: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        validate_args: bool = False,
    ):
        super().__init__(
            total_count=total_count,
            probs=probs,
            logits=logits,
            mu=mu,
            theta=theta,
            scale=scale,
            validate_args=validate_args,
        )
        self.zi_logits, self.mu, self.theta = broadcast_all(
            zi_logits, self.mu, self.theta
        )

    @property
    def mean(self) -> torch.Tensor:
        return (1 - self.zi_probs) * self.mu

    @property
    def variance(self) -> torch.Tensor:
        pi = self.zi_probs
        return (
            (1 - pi) * self.mu * (self.mu + self.theta + pi * self.mu * self.theta)
            / self.theta
        )

    @property
    def zi_probs(self) -> torch.Tensor:
        return torch.sigmoid(self.zi_logits)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        try:
            self._validate_sample(value)
        except ValueError:
            warnings.warn(
                "The value argument must be within the support of the distribution",
                UserWarning,
                stacklevel=2,
            )
        return log_zinb_positive(
            value, self.mu, self.theta, self.zi_logits, eps=1e-08
        )
