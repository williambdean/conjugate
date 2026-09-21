import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def matplotib_cleanup():
    yield
    plt.close("all")
