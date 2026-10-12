import numpy as np
import pytest
from matplotlib.axes import Axes

from conjugate.distributions import Beta, Binomial
from conjugate.plot import resolve_label


@pytest.mark.parametrize(
    "label, yy, expected",
    [
        ("label", np.array([1, 2, 3]), "label"),
        (None, np.array([1, 2, 3]), None),
        (["label1", "label2"], np.array([1, 2]), ["label1", "label2"]),
        ("label", np.ones(shape=(3, 2)), ["label 1", "label 2"]),
        ("label", np.ones(shape=(2, 3)), ["label 1", "label 2", "label 3"]),
        (
            lambda i: f"another {i + 1} label",
            np.ones(shape=(2, 3)),
            ["another 1 label", "another 2 label", "another 3 label"],
        ),
    ],
)
def test_resolve_label(label, yy, expected):
    assert resolve_label(label, yy) == expected


def test_plot_pdf() -> None:
    dist = Beta(alpha=1, beta=1)
    ax = dist.plot_pdf()
    assert isinstance(ax, Axes)


def test_plot_cdf() -> None:
    dist = Beta(alpha=1, beta=1)
    ax = dist.plot_cdf()
    assert isinstance(ax, Axes)


def test_plot_pmf() -> None:
    dist = Binomial(n=10, p=0.5)
    ax = dist.plot_pmf()
    assert isinstance(ax, Axes)
