---
title: 'conjugate-models: Conjugate Models in Python'
tags:
  - Python
  - Bayesian statistics
  - conjugate priors
  - machine learning
  - data science
authors:
  - name: William Dean
    orcid: 0009-0003-6510-3545
    affiliation: 1
affiliations:
 - name: Independent Researcher
   index: 1
date: 29 December 2025
bibliography: paper.bib
---

# Summary

`conjugate-models` is a modern Python package for Bayesian conjugate inference that prioritizes a clean, idiomatic API and seamless integration with widely used Python data analysis libraries. It covers the majority of common conjugate likelihood-prior pairs found in statistical literature, making rigorous Bayesian updating, exploration, and visualization accessible for practitioners, educators, and researchers. Comprehensive documentation, interactive examples, and an online Distribution Explorer further support flexible application and learning.

# Statement of need

Bayesian inference with conjugate priors offers a tractable and interpretable
approach to updating distributions in light of new evidence. While
general-purpose probabilistic programming frameworks exist, they can introduce
significant cognitive and computational overhead for common conjugate models. The `conjugate-models`
package is designed to provide efficient, intuitive, and didactic support for
Bayesian conjugate workflows across statistics, data science, and education,
allowing direct use with array-like objects from libraries such as
numpy[@harris2020array],
pandas[@The_pandas_development_team_pandas-dev_pandas_Pandas], and
polars[@polars2024]. The project also provides live interactive
documentation and an [online Distribution Explorer](https://williambdean.github.io/conjugate/explorer) for real-time model investigation.

## Conjugate Priors

A prior distribution is conjugate to a likelihood when the posterior remains in
the same distribution family after observing data[@raiffa1961applied].
Conjugate priors provide closed-form posterior updates and posterior predictive
distributions, eliminating the need for numerical
integration[@fink1997compendium]. Classic examples include the
Beta-Binomial, Gamma-Poisson, and Normal-Normal families, each of which admits
compact updates for parameters such as success probability, rate, or
mean[@murphy2007conjugate]. `conjugate-models` implements the majority of
conjugate pairs cataloged in the literature and reference
tables[@fink1997compendium].

## Motivation

Many real-world and educational Bayesian inference tasks benefit from the
simplicity and tractability of conjugate models. However, tooling for these
models in Python has lagged behind full-featured—and often
complex—probabilistic frameworks. `conjugate-models` fills this gap for
users seeking:
- A composable API that interoperates smoothly with scientific Python libraries (numpy, pandas, polars, matplotlib, and others)
- Minimal cognitive overhead and easy expression of classic Bayesian updates
- Coverage of most conjugate prior-likelihood pairs outlined in statistical literature and on the [Wikipedia Conjugate Prior page](https://en.wikipedia.org/wiki/Conjugate_prior)
- Interactive and educational resources, including an [online Distribution Explorer](https://williambdean.github.io/conjugate/explorer)

## Problem Statement

Python lacked a dedicated, user-friendly package making Bayesian conjugate
inference accessible, idiomatic, and didactically powerful—especially one
offering robust integration with common data tooling and interactive resources
for teaching and exploratory work.

# Features & Capabilities
`conjugate-models` provides:
- An intuitive, pipeable API compatible with numpy arrays[@harris2020array], pandas DataFrames/Series[@The_pandas_development_team_pandas-dev_pandas_Pandas], polars DataFrames[@polars2024] (for element-wise operations), and general numerical types
- Vectorized and indexable operations for batch and multi-arm inference
- Built-in plotting for posterior, prior, and predictive distributions
- Connection to scipy distributions for interoperability[@virtanen2020scipy]
- Support for nearly all likelihood-prior pairs listed in statistical literature and Wikipedia
- An [interactive Distribution Explorer](https://williambdean.github.io/conjugate/explorer) and live, documented [examples and use cases](https://williambdean.github.io/conjugate/examples/)

## API Overview

A typical workflow follows a consistent pattern that keeps imports, summary
statistics, and posterior updates easy to reason about:

```python
from conjugate.distributions import SomeDistribution
from conjugate.models import some_model

prior: SomeDistribution = ...
data = ...
summary_stats = f(data)
posterior: SomeDistribution = some_model(*summary_stats, prior=prior)
```

Predictive distributions are exposed alongside analytic posteriors when
conjugate forms exist: the model computed above will also have a
`some_model_predictive` companion. When closed-form predictives are not
available, posterior random samples can be drawn and fed through the
likelihood; see
[the unsupported distributions example](https://williambdean.github.io/conjugate/examples/unsupported-distributions) for a detailed demonstration.

This structure fits naturally within the broader Bayesian
workflow[@gelman2020workflow]: specifying a prior, summarizing data,
updating to a posterior, generating posterior predictive checks, and iterating
as needed.

## API Generality and Extensibility

The API is designed for compatibility with a broad array of array-like or
DataFrame-like objects—numpy, pandas, polars, and more—allowing seamless
integration in scientific Python workflows. The package covers the majority of
conjugate pairs commonly found in the literature, and its general design makes
extension to new models straightforward. Learn more and browse the full list of
supported models at the
[documentation](https://williambdean.github.io/conjugate/models/).

# Example Usage

## Sequential Bayesian Updates

The library supports incremental Bayesian updating where the posterior from one
batch becomes the prior for the next.

Abstractly, the workflow looks like this:

```python
prior = ...

for batch_size in batch_sizes:
    data = sample(n=batch_size)
    posterior = model(data, prior=prior)
    prior = posterior
```

The normal likelihood with a `NormalInverseGamma` prior offers a concrete illustration:

```python
import numpy as np
from conjugate.distributions import NormalInverseGamma
from conjugate.models import normal

rng = np.random.default_rng(0)
sample = lambda n: rng.normal(loc=5.0, scale=2.5, size=n)

prior = NormalInverseGamma(mu=0, alpha=1, beta=1, nu=1)
for batch_size in [5, 10, 25]:
    data = sample(batch_size)
    posterior = normal(
        x_total=data.sum(),
        x2_total=(data**2).sum(),
        n=batch_size,
        prior=prior,
    )
    prior = posterior
```

Posterior samples or plots reveal how uncertainty shrinks across batches,
demonstrating the canonical “posterior-as-new-prior” flow. A full walkthrough
with visualizations is available in the
[Bayesian update example](https://williambdean.github.io/conjugate/examples/bayesian-update).

![Sequential posterior samples for the normal-normal model as new data arrive.](../docs/images/bayesian-update.png)


## Bayesian Linear Regression

For regression tasks, `conjugate-models` handles the multivariate
Normal-Inverse-Gamma hierarchy, offering a Bayesian alternative to tools like
`scikit-learn`'s `BayesianRidge` by yielding full posterior distributions on
coefficients and variance:

```python
import numpy as np
from conjugate.distributions import NormalInverseGamma
from conjugate.models import linear_regression

# Design matrix X (with intercept) and target y
X = np.array([[1, 1.5], [1, 2.5], [1, 3.5]])
y = np.array([4.0, 6.0, 8.0])

prior = NormalInverseGamma(
    mu=np.zeros(2),
    alpha=1.0,
    beta=1.0,
    delta_inverse=np.eye(2),
)

posterior = linear_regression(X=X, y=y, prior=prior)
```

## Beta–Binomial Across Data Backends

A single API call works regardless of whether observations arrive as native
Python numbers, NumPy arrays, pandas Series, or Polars Series. The following
example updates the same Beta–Binomial model across each backend without
changing the model invocation:

```python
from conjugate.distributions import Beta
from conjugate.models import binomial_beta, binomial_beta_predictive
import numpy as np
import pandas as pd
import polars as pl

# Data: 4 successes in 10 trials
x = 4
N = 10

prior = Beta(1, 1)

# builtin
posterior: Beta = binomial_beta(n=N, x=x, prior=prior)
# numpy
np_posterior: Beta = binomial_beta(n=N, x=np.array([x]), prior=prior)
# pandas
pd_posterior: Beta = binomial_beta(n=N, x=pd.Series([x]), prior=prior)
# polars
pl_posterior: Beta = binomial_beta(n=N, x=pl.Series([x]), prior=prior)

# Plotting (identical regardless of backend)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
np_posterior.plot_pdf(ax=ax, label="posterior (numpy)")
ax.legend()
plt.show()
```

![Posterior and Posterior Predictive for Binomial-Beta Model.](../docs/images/binomial-beta.png)

More backend demonstrations and notebooks are available in the
[vectorized inputs example](https://williambdean.github.io/conjugate/examples/vectorized-inputs) and the
[Distribution Explorer](https://williambdean.github.io/conjugate/explorer).

## Thompson Sampling with Multi-Armed Bandits

Vectorized operations make it straightforward to maintain and update posterior
beliefs over many arms simultaneously, allowing planners to rescore all arms in
one pass:

```python
from conjugate.distributions import Beta
from conjugate.models import binomial_beta
import numpy as np

# Three arms, uniform priors
prior = Beta(1, 1)

# Observed successes and trials per arm (vectorized input)
successes = np.array([2, 3, 5])
trials = np.array([5, 6, 10])

# Vectorized posterior update for all arms
posterior = binomial_beta(n=trials, x=successes, prior=prior)

# Thompson Sampling: sample all posterior at once
samples = posterior.dist.rvs()
chosen_arm = int(np.argmax(samples))
print(f"Thompson Sampling selects arm: {chosen_arm}")
```

![Posterior samples and action selection for a multi-armed Thompson sampling bandit.](../docs/images/thompson.png)

Extended and interactive Thompson Sampling examples are available
[here](https://williambdean.github.io/conjugate/examples/thompson).

# Related Work

Several libraries offer Bayesian modeling, including PyMC[@pymc2023],
Stan/Pystan[@stan2025reference], and
scipy.stats[@virtanen2020scipy]. These tools provide immense generality
but are typically heavyweight for conjugate cases, prioritizing MCMC and
broader, non-conjugate models. `conjugate-models` distinguishes itself by
focusing on a broad and faithful implementation of all classical conjugate
priors in an accessible API tailored for Python users, educators, and
researchers who prefer clarity, rapid analysis, integration with common
scientific ecosystems, and interactive documentation. Its didactic approach and
extensibility make it an appealing alternative and companion to more complex
probabilistic frameworks.
