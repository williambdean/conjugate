import pytest

from conjugate.distributions import Beta
from conjugate.models import binomial_beta, bernoulli_beta
from conjugate.interactive import (
    get_prior_distribution,
    lookup_model,
    lookup_model_by_predictive,
)


@pytest.mark.parametrize(
    "func",
    [
        binomial_beta,
        bernoulli_beta,
    ],
)
def test_get_prior_distribution(func):
    assert get_prior_distribution(func) == Beta


@pytest.mark.parametrize(
    "likelihood, expected",
    [
        ("Binomial", "binomial_beta"),
        ("Geometric", "geometric_beta"),
        ("Gamma", ["gamma_known_shape", "gamma", "gamma_known_rate"]),
    ],
)
def test_lookup_model(likelihood, expected):
    assert set(lookup_model(likelihood)) == set(expected)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("BetaBinomial", ["bernoulli_beta", "binomial_beta"]),
        ("BetaGeometric", "geometric_beta"),
    ],
)
def test_lookup_predictive(name, expected) -> None:
    assert lookup_model_by_predictive(name) == expected
