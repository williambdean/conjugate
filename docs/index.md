---
hide:
    - navigation
---
# Conjugate Models

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://github.com/williambdean/conjugate/actions/workflows/tests.yml/badge.svg)](https://github.com/williambdean/conjugate/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/conjugate-models.svg)](https://badge.fury.io/py/conjugate-models)
[![docs](https://github.com/williambdean/conjugate/actions/workflows/docs.yml/badge.svg)](https://williambdean.github.io/conjugate/)
[![codecov](https://codecov.io/github/williambdean/conjugate/branch/main/graph/badge.svg)](https://app.codecov.io/github/williambdean/conjugate)

Bayesian conjugate models in Python

## Overview

`conjugate-models` is a modern Python package for Bayesian conjugate inference that prioritizes a clean, idiomatic API and seamless integration with widely used Python data analysis libraries. It implements the conjugate likelihood-prior pairs cataloged in [Fink's compendium](https://www.johndcook.com/CompendiumOfConjugatePriors.pdf) and [Wikipedia's conjugate prior table](https://en.wikipedia.org/wiki/Conjugate_prior), making rigorous Bayesian updating, exploration, and visualization accessible for practitioners, educators, and researchers.

### Why Conjugate Priors?

A prior distribution is conjugate to a likelihood when the posterior remains in the same distribution family after observing data. Conjugate priors provide closed-form posterior updates and posterior predictive distributions, eliminating the need for numerical integration or MCMC sampling. Because these updates are analytic rather than iterative, **posterior computation is instantaneous regardless of data size**—enabling real-time interactive exploration and rapid model iteration.

### Key Benefits

- ⚡ **Instant Updates:** No MCMC or optimization required—posterior computation is immediate
- 🔢 **Vectorized Operations:** Batch inference for multi-arm problems without explicit loops
- 📊 **Built-in Visualization:** Plot priors, posteriors, and predictive distributions
- 🔗 **SciPy Integration:** Direct access to scipy.stats distributions via `.dist` property
- 📦 **Data Library Support:** Works seamlessly with numpy, pandas, polars, and general array-like objects
- 🪶 **Lightweight Dependencies:** Minimal requirements—no heavy ML frameworks or complex toolchains

### Lightweight & Easy to Install

With minimal dependencies from the scientific Python stack, `conjugate-models` installs quickly without requiring heavyweight probabilistic programming frameworks, MCMC samplers, or complex compilation toolchains.

## Installation

```bash
pip install conjugate-models
```

## Features

- [Interactive Distribution Explorer](explorer.md) for exploring probability distributions with real-time parameter adjustment *(temporarily unavailable — see [#324](https://github.com/williambdean/conjugate/issues/324))*
- **[Raw Data Workflow](examples/raw-data-workflow.md)** - Complete examples from raw observational data to posterior distributions with helper functions
- **[Data Input Helper Functions](helpers.md)** - Extract sufficient statistics from raw observational data for all supported models
- [Connection to Scipy Distributions](examples/scipy-connection.md) with `dist` attribute
- [Built in Plotting](examples/plotting.md) with `plot_pdf`, `plot_pmf`, and `plot_cdf` methods
- [Vectorized Operations](examples/vectorized-inputs.md) for parameters and data
- [Indexing Parameters](examples/indexing.md) for subsetting and slicing
- [Generalized Numerical Inputs](examples/generalized-inputs.md) for any inputs that act like numbers
    - Out of box compatibility with `polars`, `pandas`, `numpy`, and more.
- [Unsupported Distributions](examples/unsupported-distributions.md) for sampling from unsupported distributions

## Supported Models

Many likelihoods are supported including

- `Bernoulli` / `Binomial`
- `Categorical` / `Multinomial`
- `Poisson`
- `Normal` (including linear regression)
- and [many more](models.md)

See the [Quick Reference](quick-reference.md) for a complete table of likelihood → prior/posterior mappings with links to model functions and helper functions.

## Basic Usage

### Pattern 1: Working with Pre-processed Data

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

### Pattern 2: Working with Raw Observational Data

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
# Returns: {'x_total': sum(count_data), 'n': len(count_data)}

# Continuous measurements with known variance
measurements = [2.3, 1.9, 2.7, 2.1, 2.5]
inputs = normal_known_variance_inputs(measurements)
# Returns: {'x_total': sum(measurements), 'n': len(measurements)}
# Note: variance must be passed separately to the model function

# Time between events (e.g., customer arrivals)
wait_times = [3.2, 1.8, 4.1, 2.7, 3.9]
inputs = exponential_gamma_inputs(wait_times)
# Returns: {'x_total': sum(wait_times), 'n': len(wait_times)}

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

<img height=400 src="images/binomial-beta.png" title="Binomial Beta Comparison">

## Too Simple?

Simple model, sure. Useful model, potentially.

Constant probability of success, `p`, for `n` trials.

```python
rng = np.random.default_rng(42)

# Observed Data
n_times = 75
p = np.repeat(0.5, n_times)
samples = rng.binomial(n=1, p=p, size=n_times)

# Model
n = np.arange(n_times) + 1
prior = Beta(alpha=1, beta=1)
posterior = binomial_beta(n=n, x=samples.cumsum(), prior=prior)

# Figure
plt.plot(n, p, color="black", label="true p", linestyle="--")
plt.scatter(n, samples, color="black", label="observed samples")
plt.plot(n, posterior.dist.mean(), color="red", label="posterior mean")
# fill between the 95% credible interval
plt.fill_between(
    n,
    posterior.dist.ppf(0.025),
    posterior.dist.ppf(0.975),
    color="red",
    alpha=0.2,
    label="95% credible interval",
)
padding = 0.025
plt.ylim(0 - padding, 1 + padding)
plt.xlim(1, n_times)
plt.legend(loc="best")
plt.xlabel("Number of trials")
plt.ylabel("Probability")
plt.show()
```

<img height=400 src="images/constant-probability.png" title="Constant Probability">

Even with a moving probability, this simple to implement model can be useful.

```python
...


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


p_raw = rng.normal(loc=0, scale=0.2, size=n_times).cumsum()
p = sigmoid(p_raw)

...
```


<img height=400 src="images/moving-probability.png" title="Moving Probability">


## Resources

- [Conjugate Priors](https://en.wikipedia.org/wiki/Conjugate_prior)
