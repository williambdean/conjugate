import pytest

import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def matplotib_cleanup():
    yield
    plt.close("all")
