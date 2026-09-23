"""Tests for vendored NegativeBinomial and ZeroInflatedNegativeBinomial distributions."""

import pytest
import torch

from celltransformer.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial


# ---------------------------------------------------------------------------
# NegativeBinomial
# ---------------------------------------------------------------------------


class TestNegativeBinomial:
    def test_construction_mu_theta(self):
        # (mu, theta) parameterization accepts valid inputs
        d = NegativeBinomial(mu=torch.tensor([1.0]), theta=torch.tensor([1.0]))
        assert d.mu is not None and d.theta is not None

    def test_construction_total_count_probs(self):
        # alternative (total_count, probs) parameterization also works
        d = NegativeBinomial(total_count=torch.tensor([5.0]), probs=torch.tensor([0.3]))
        assert d.mu is not None

    def test_construction_both_raises(self):
        # supplying both parameterizations is ambiguous and should error (mu/theta/count and probs)
        with pytest.raises(ValueError):
            NegativeBinomial(
                mu=torch.tensor([1.0]),
                theta=torch.tensor([1.0]),
                total_count=torch.tensor([5.0]),
                probs=torch.tensor([0.3]),
            )

    def test_construction_neither_raises(self):
        # supplying neither parameterization should error
        with pytest.raises(ValueError):
            NegativeBinomial()
            
    def test_log_prob_finite(self):
        # no NaN/Inf for typical inputs
        d = NegativeBinomial(mu=torch.tensor([1.0, 5.0, 10.0]), theta=torch.tensor([0.5, 1.0, 2.0]))
        lp = d.log_prob(torch.tensor([0.0, 3.0, 7.0]))
        assert torch.isfinite(lp).all()

    def test_log_prob_non_positive(self):
        # log-probabilities must be <= 0
        d = NegativeBinomial(mu=torch.tensor([1.0, 5.0]), theta=torch.tensor([1.0, 5.0]))
        lp = d.log_prob(torch.tensor([0.0, 10.0]))
        assert (lp <= 1e-6).all()  # small tolerance for numerical noise

    def test_log_prob_zero_vs_large(self):
        # for small mu, P(x=0) should dominate P(x=100)
        d = NegativeBinomial(mu=torch.tensor([0.5]), theta=torch.tensor([1.0]))
        lp_zero = d.log_prob(torch.tensor([0.0]))
        lp_large = d.log_prob(torch.tensor([100.0]))
        assert lp_zero > lp_large

    def test_mean(self):
        # mean property should equal mu
        mu = torch.tensor([1.0, 3.0, 5.0])
        d = NegativeBinomial(mu=mu, theta=torch.tensor([2.0, 2.0, 2.0]))
        assert torch.allclose(d.mean, mu)

    def test_variance(self):
        # variance = mu + mu^2 / theta
        mu = torch.tensor([2.0, 4.0])
        theta = torch.tensor([1.0, 3.0])
        d = NegativeBinomial(mu=mu, theta=theta)
        expected = mu + mu**2 / theta
        assert torch.allclose(d.variance, expected)

    def test_broadcasting_scalar_theta(self):
        # scalar theta broadcasts across batched mu
        mu = torch.tensor([1.0, 2.0, 3.0])
        theta = torch.tensor([1.0])
        d = NegativeBinomial(mu=mu, theta=theta)
        lp = d.log_prob(torch.zeros(3))
        assert lp.shape == (3,)
        assert torch.isfinite(lp).all()


# ---------------------------------------------------------------------------
# ZeroInflatedNegativeBinomial
# ---------------------------------------------------------------------------


class TestZeroInflatedNegativeBinomial:
    def test_construction(self):
        # (mu, theta, zi_logits) parameterization accepts valid inputs
        d = ZeroInflatedNegativeBinomial(
            mu=torch.tensor([1.0]), theta=torch.tensor([1.0]), zi_logits=torch.tensor([0.0])
        )
        assert d.mu is not None and d.zi_logits is not None

    def test_log_prob_shape(self):
        # 2D inputs preserve shape exactly
        d = ZeroInflatedNegativeBinomial(
            mu=torch.tensor([[1.0, 2.0]]),
            theta=torch.tensor([[1.0, 2.0]]),
            zi_logits=torch.tensor([[0.0, 0.0]]),
        )
        x = torch.tensor([[0.0, 5.0]])
        assert d.log_prob(x).shape == x.shape

    def test_log_prob_shape_1d_theta_broadcast(self):
        # 1D theta gets reshaped to [1, N] inside log_zinb_positive (scvi convention
        # for gene-level dispersion), so output gains a batch dim
        d = ZeroInflatedNegativeBinomial(
            mu=torch.tensor([1.0, 2.0]),
            theta=torch.tensor([1.0, 2.0]),
            zi_logits=torch.tensor([0.0, 0.0]),
        )
        lp = d.log_prob(torch.tensor([0.0, 5.0]))
        assert lp.shape == (1, 2)

    def test_log_prob_finite(self):
        # no NaN/Inf for typical inputs
        d = ZeroInflatedNegativeBinomial(
            mu=torch.tensor([1.0, 5.0]),
            theta=torch.tensor([0.5, 2.0]),
            zi_logits=torch.tensor([0.0, -1.0]),
        )
        lp = d.log_prob(torch.tensor([0.0, 3.0]))
        assert torch.isfinite(lp).all()

    def test_log_prob_non_positive(self):
        # log-probabilities must be <= 0
        d = ZeroInflatedNegativeBinomial(
            mu=torch.tensor([1.0, 5.0]),
            theta=torch.tensor([1.0, 5.0]),
            zi_logits=torch.tensor([0.0, 0.0]),
        )
        lp = d.log_prob(torch.tensor([0.0, 10.0]))
        assert (lp <= 1e-6).all()

    def test_zero_inflation_effect(self):
        # more dropout (high zi_logits) should increase P(x=0)
        mu = torch.tensor([5.0])
        theta = torch.tensor([1.0])
        x = torch.tensor([0.0])

        d_high_zi = ZeroInflatedNegativeBinomial(mu=mu, theta=theta, zi_logits=torch.tensor([5.0]))
        d_low_zi = ZeroInflatedNegativeBinomial(mu=mu, theta=theta, zi_logits=torch.tensor([-5.0]))

        assert d_high_zi.log_prob(x) > d_low_zi.log_prob(x)

    def test_reduces_to_nb(self):
        # zi_logits=-20 effectively disables zero inflation, should match plain NB
        mu = torch.tensor([1.0, 3.0, 5.0])
        theta = torch.tensor([2.0, 2.0, 2.0])
        x = torch.tensor([0.0, 2.0, 4.0])

        nb = NegativeBinomial(mu=mu, theta=theta)
        zinb = ZeroInflatedNegativeBinomial(mu=mu, theta=theta, zi_logits=torch.tensor([-20.0]))

        assert torch.allclose(zinb.log_prob(x), nb.log_prob(x), atol=1e-4)

    def test_mean(self):
        # mean = (1 - sigmoid(zi_logits)) * mu
        mu = torch.tensor([2.0, 4.0])
        zi_logits = torch.tensor([0.0, 1.0])
        d = ZeroInflatedNegativeBinomial(mu=mu, theta=torch.tensor([1.0, 1.0]), zi_logits=zi_logits)
        expected = (1 - torch.sigmoid(zi_logits)) * mu
        assert torch.allclose(d.mean, expected)

    def test_zi_probs(self):
        # zi_probs property should equal sigmoid(zi_logits)
        zi_logits = torch.tensor([-2.0, 0.0, 2.0])
        d = ZeroInflatedNegativeBinomial(
            mu=torch.ones(3), theta=torch.ones(3), zi_logits=zi_logits
        )
        assert torch.allclose(d.zi_probs, torch.sigmoid(zi_logits))


# ---------------------------------------------------------------------------
# Numerical edge cases (shared)
# ---------------------------------------------------------------------------


class TestNumericalEdgeCases:
    @pytest.mark.parametrize("dist_cls,extra_kw", [
        (NegativeBinomial, {}),
        (ZeroInflatedNegativeBinomial, {"zi_logits": torch.tensor([0.0])}),
    ])
    def test_very_small_mu_theta(self, dist_cls, extra_kw):
        # near-eps parameters should not produce NaN
        d = dist_cls(mu=torch.tensor([1e-8]), theta=torch.tensor([1e-8]), **extra_kw)
        lp = d.log_prob(torch.tensor([0.0]))
        assert torch.isfinite(lp).all()

    @pytest.mark.parametrize("dist_cls,extra_kw", [
        (NegativeBinomial, {}),
        (ZeroInflatedNegativeBinomial, {"zi_logits": torch.tensor([0.0])}),
    ])
    def test_very_large_mu_theta(self, dist_cls, extra_kw):
        # large parameters should not overflow to Inf
        d = dist_cls(mu=torch.tensor([1e6]), theta=torch.tensor([1e6]), **extra_kw)
        lp = d.log_prob(torch.tensor([100.0]))
        assert torch.isfinite(lp).all()

    @pytest.mark.parametrize("dist_cls,extra_kw", [
        (NegativeBinomial, {}),
        (ZeroInflatedNegativeBinomial, {"zi_logits": torch.tensor([0.0, 0.0, 0.0])}),
    ])
    def test_all_zero_counts(self, dist_cls, extra_kw):
        # all-zero observed counts are common in scRNA-seq, must be finite
        d = dist_cls(mu=torch.tensor([1.0, 2.0, 3.0]), theta=torch.tensor([1.0, 2.0, 3.0]), **extra_kw)
        lp = d.log_prob(torch.zeros(3))
        assert torch.isfinite(lp).all()
