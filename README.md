# Conjugate Models

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://github.com/williambdean/conjugate/actions/workflows/tests.yml/badge.svg)](https://github.com/williambdean/conjugate/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/conjugate-models.svg)](https://badge.fury.io/py/conjugate-models)
[![docs](https://github.com/williambdean/conjugate/actions/workflows/docs.yml/badge.svg)](https://williambdean.github.io/conjugate/)
[![codecov](https://codecov.io/github/williambdean/conjugate/branch/main/graph/badge.svg)](https://app.codecov.io/github/williambdean/conjugate)

Bayesian conjugate models in Python


## Installation

```bash
pip install conjugate-models
```

## Features

- [Interactive Distribution Explorer](https://williambdean.github.io/conjugate/explorer) for exploring probability distributions with real-time parameter adjustment
- **[Raw Data Workflow](https://williambdean.github.io/conjugate/examples/raw-data-workflow)** - Complete examples from raw observational data to posterior distributions with helper functions
- **Data Input Helper Functions** - Extract sufficient statistics from raw observational data for all supported models
- [Connection to Scipy Distributions](https://williambdean.github.io/conjugate/examples/scipy-connection) with `dist` attribute
- [Built in Plotting](https://williambdean.github.io/conjugate/examples/plotting) with `plot_pdf`, `plot_pmf`, and `plot_cdf` methods
- [Vectorized Operations](https://williambdean.github.io/conjugate/examples/vectorized-inputs) for parameters and data
- [Indexing Parameters](https://williambdean.github.io/conjugate/examples/indexing) for subsetting and slicing
- [Generalized Numerical Inputs](https://williambdean.github.io/conjugate/examples/generalized-inputs) for any inputs that act like numbers
    - Out of box compatibility with `polars`, `pandas`, `numpy`, and more.
- [Unsupported Distributions](https://williambdean.github.io/conjugate/examples/unsupported-distributions) for sampling from unsupported distributions

## Supported Models

Many likelihoods are supported including

- `Bernoulli` / `Binomial`
- `Categorical` / `Multinomial`
- `Poisson`
- `Normal` (including linear regression)
- and [many more](https://williambdean.github.io/conjugate/models/)

## Basic Usage

### Working with Pre-processed Data

1. Define prior distribution from `distributions` module
1. Pass data and prior into model from `models` modules
1. Analytics with posterior and posterior predictive distributions

```python
from conjugate.distributions import Beta, BetaBinomial
from conjugate.models import binomial_beta, binomial_beta_predictive

# Observed Data (sufficient statistics)
x = 4  # successes
N = 10 # trials

# Analytics
prior = Beta(1, 1)
prior_predictive: BetaBinomial = binomial_beta_predictive(n=N, distribution=prior)

posterior: Beta = binomial_beta(n=N, x=x, prior=prior)
posterior_predictive: BetaBinomial = binomial_beta_predictive(
    n=N, distribution=posterior
)
```

### Working with Raw Observational Data

For raw data, use **helper functions** from the `helpers` module to extract sufficient statistics:

```python
import numpy as np
from conjugate.distributions import Beta
from conjugate.models import binomial_beta
from conjugate.helpers import bernoulli_beta_inputs

# Raw observational data - individual trial outcomes
raw_data = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]  # success/failure per trial

# Extract sufficient statistics automatically
inputs = bernoulli_beta_inputs(raw_data)
print(inputs)  # {'x': 6, 'n': 10} - 6 successes in 10 trials

# Use with conjugate model
prior = Beta(1, 1)
posterior = binomial_beta(prior=prior, **inputs)
```

#### Common Helper Function Patterns

```python
from conjugate.helpers import (
    poisson_gamma_inputs,      # For count data
    normal_known_variance_inputs,  # For continuous measurements
    exponential_gamma_inputs,  # For time-between-events data
    multinomial_dirichlet_inputs,  # For categorical data
)

# Count data (e.g., website visits per day)
count_data = [5, 3, 8, 2, 6, 4, 7, 1, 9, 3]
inputs = poisson_gamma_inputs(count_data)
# Returns: {'x': sum(count_data), 'n': len(count_data)}

# Continuous measurements with known variance
measurements = [2.3, 1.9, 2.7, 2.1, 2.5]
variance = 0.5
inputs = normal_known_variance_inputs(measurements, variance=variance)
# Returns: {'x_mean': mean(measurements), 'n': len(measurements), 'variance': variance}

# Time between events (e.g., customer arrivals)
wait_times = [3.2, 1.8, 4.1, 2.7, 3.9]
inputs = exponential_gamma_inputs(wait_times)
# Returns: {'x': sum(wait_times), 'n': len(wait_times)}

# Categorical outcomes (e.g., survey responses A, B, C)
responses = ['A', 'B', 'A', 'C', 'B', 'A', 'B']
inputs = multinomial_dirichlet_inputs(responses)
# Returns: {'x': [3, 3, 1]} - counts for each category
```

All 50+ helper functions follow the same pattern: **raw observations in → sufficient statistics out** → ready for conjugate models.

From here, do any analysis you'd like!

```python
# Figure
import matplotlib.pyplot as plt

fig, axes = plt.subplots(ncols=2)

ax = axes[0]
ax = posterior.plot_pdf(ax=ax, label="posterior")
prior.plot_pdf(ax=ax, label="prior")
ax.axvline(x=x / N, color="black", ymax=0.05, label="MLE")
ax.set_title("Success Rate")
ax.legend()

ax = axes[1]
posterior_predictive.plot_pmf(ax=ax, label="posterior predictive")
prior_predictive.plot_pmf(ax=ax, label="prior predictive")
ax.axvline(x=x, color="black", ymax=0.05, label="Sample")
ax.set_title("Number of Successes")
ax.legend()
plt.show()
```

<img height=400 src="docs/images/binomial-beta.png" title="Binomial Beta Comparison">

More examples on in the [documentation](https://williambdean.github.io/conjugate/).

## Contributing

If you are interested in contributing, check out the [contributing guidelines](https://github.com/williambdean/conjugate/blob/main/CONTRIBUTING.md)
