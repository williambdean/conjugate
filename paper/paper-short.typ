#import "conf.typ": conf

#show link: set text(hyphenate: true, fill: rgb("#1f77b4").lighten(20%))

#show: conf.with(
  title: [`conjugate-models`: Conjugate Models in Python],
  authors: (
    (
      name: "William Dean",
      email: "wd60622@gmail.com",
    ),
  ),
  abstract: [
    `conjugate-models` is a modern Python package for Bayesian conjugate
    inference that prioritizes a clean, idiomatic API and seamless integration
    with widely used Python data analysis libraries. It implements the conjugate
    likelihood-prior pairs cataloged in Fink's compendium and Wikipedia's
    conjugate prior table, making rigorous Bayesian updating, exploration, and
    visualization accessible for practitioners, educators, and researchers.
    Comprehensive documentation, interactive examples, and an online Distribution
    Explorer further support flexible application and learning.
  ],
)

= Introduction

Bayesian inference with conjugate priors offers a tractable and interpretable
approach to updating distributions in light of new evidence. While
general-purpose probabilistic programming frameworks exist, they can introduce
significant cognitive and computational overhead for common conjugate models. The `conjugate-models`
package is designed to provide efficient, intuitive, and didactic support for
Bayesian conjugate workflows across statistics, data science, and education,
allowing direct use with array-like objects from libraries such as
numpy#cite(<harris2020array>),
pandas#cite(<The_pandas_development_team_pandas-dev_pandas_Pandas>), and
polars#cite(<polars2024>). The project also provides live interactive
documentation and an #link("https://williambdean.github.io/conjugate/explorer",
"online Distribution Explorer") for real-time model investigation.

A prior distribution is conjugate to a likelihood when the posterior remains in
the same distribution family after observing data#cite(<raiffa1961applied>).
Conjugate priors provide closed-form posterior updates and posterior predictive
distributions, eliminating the need for numerical
integration#cite(<fink1997compendium>). Because these updates are analytic
rather than iterative, posterior computation is instantaneous regardless of
data size—enabling real-time interactive exploration and rapid model
iteration. `conjugate-models` implements the conjugate pairs cataloged in Fink's
compendium#cite(<fink1997compendium>) and Wikipedia's conjugate prior table.
The complete list of supported models is maintained at
#link("https://williambdean.github.io/conjugate/models/", "the online documentation").

Python lacked a dedicated, user-friendly package making Bayesian conjugate
inference accessible, idiomatic, and didactically powerful—especially one
offering robust integration with common data tooling and interactive resources
for teaching and exploratory work. Existing educational tools for Bayesian
statistics often lack interactivity or require complex setup, making the
intuitive nature of conjugate priors difficult to convey to students and
practitioners new to Bayesian methods.

= Features & API Overview

`conjugate-models` provides an intuitive, pipeable API compatible with numpy arrays#cite(<harris2020array>), pandas DataFrames/Series#cite(<The_pandas_development_team_pandas-dev_pandas_Pandas>), polars DataFrames#cite(<polars2024>) (for element-wise operations), and general numerical types. The package includes vectorized and indexable operations for batch and multi-arm inference, built-in plotting for posterior, prior, and predictive distributions, and connection to scipy distributions for interoperability#cite(<virtanen2020scipy>).

A typical workflow follows a consistent pattern:

```python
from conjugate.distributions import SomeDistribution
from conjugate.models import some_model

prior: SomeDistribution = ...
data = ...
summary_stats = f(data)
posterior: SomeDistribution = some_model(*summary_stats, prior=prior)
```

= Example Usage

== Sequential Bayesian Updates

The package naturally supports sequential Bayesian learning, where each
posterior becomes the next prior. This example demonstrates real-time updating
for a binomial model with beta prior:

```python
from conjugate.distributions import Beta
from conjugate.models import binomial_beta

# Initial prior: uniform over [0, 1]
prior = Beta(alpha=1, beta=1)

# Observations: 3 successes in 5 trials, then 2 successes in 3 trials
observations = [(3, 5), (2, 3)]

posterior = prior
for successes, trials in observations:
    posterior = binomial_beta(n=trials, x=successes, prior=posterior)
    print(f"After {successes}/{trials}: α={posterior.alpha}, β={posterior.beta}")

# After 3/5: α=4.0, β=3.0
# After 2/3: α=6.0, β=4.0
```

#figure(
  image("bayesian-update.png", width: 85%),
  caption: [Sequential Bayesian updates showing prior, intermediate posterior, and final posterior distributions. Each update incorporates new evidence while maintaining uncertainty.]
)

== Thompson Sampling with Multi-Armed Bandits

Thompson sampling for multi-armed bandits becomes straightforward with
conjugate updates. This example shows exploration-exploitation for three
bandits with unknown success rates:

```python
import numpy as np
from conjugate.distributions import Beta
from conjugate.models import binomial_beta

# Three bandits with unknown success rates
n_bandits = 3
priors = [Beta(1, 1) for _ in range(n_bandits)]  # Uniform priors
posteriors = priors.copy()

# Thompson sampling: sample success rate from each posterior,
# choose highest, observe result, update
for round in range(100):
    # Sample from each posterior
    sampled_rates = [post.rvs(1)[0] for post in posteriors]

    # Choose bandit with highest sampled rate
    chosen_bandit = np.argmax(sampled_rates)

    # Simulate trial (e.g., true rates: [0.3, 0.5, 0.7])
    true_rates = [0.3, 0.5, 0.7]
    success = np.random.random() < true_rates[chosen_bandit]

    # Update posterior for chosen bandit
    posteriors[chosen_bandit] = binomial_beta(
        n=1, x=int(success), prior=posteriors[chosen_bandit]
    )
```

#figure(
  image("thompson.png", width: 85%),
  caption: [Thompson sampling convergence showing how posterior distributions for three bandits evolve as evidence accumulates. The algorithm learns to favor higher-reward bandits while maintaining principled exploration.]
)

= Related Work

Several Python packages address aspects of Bayesian inference.
ArviZ#cite(<Kumar2019>) excels at posterior analysis and visualization but focuses on MCMC/variational inference outputs rather than conjugate models.
Bambi#cite(<Capretto_Bambi_A_simple_2022>) provides a high-level interface for Bayesian linear models via PyMC but targets more complex model specifications.
scikit-learn#cite(<scikit-learn>) includes some Bayesian methods but emphasizes frequentist machine learning.
General probabilistic programming languages like PyMC#cite(<pymc2023>) and Stan#cite(<stan2025reference>) offer comprehensive modeling capabilities but introduce substantial overhead for simple conjugate updates.

`conjugate-models` distinguishes itself by wrapping scipy.stats distributions with conjugate update semantics, plotting interfaces, and seamless integration into standard Bayesian workflows#cite(<gelman2020workflow>). Its focused scope enables immediate posterior computation without MCMC or optimization, making it ideal for interactive exploration, education, and rapid prototyping of conjugate models.

= Limitations

This package specifically targets conjugate prior-likelihood pairs, so non-conjugate models require other tools like PyMC#cite(<pymc2023>) or Stan#cite(<stan2025reference>). Model updates operate on sufficient statistics rather than raw data, requiring users to compute summary statistics beforehand. Distribution and parameter names follow established conventions from statistical literature#cite(<fink1997compendium>), which may differ from other packages. Community support is available through GitHub Issues and Discussions.

= Acknowledgments

We thank the scientific Python community, particularly the maintainers and contributors of NumPy#cite(<harris2020array>), SciPy#cite(<virtanen2020scipy>), and Matplotlib, whose foundational libraries make this package possible. The examples showcase integration with pandas#cite(<The_pandas_development_team_pandas-dev_pandas_Pandas>), Polars#cite(<polars2024>), and PyMC#cite(<pymc2023>), demonstrating the collaborative spirit of the open-source ecosystem.

= Availability

`conjugate-models` is available on PyPI: `pip install conjugate-models`. The package is open-source under the MIT license at #link("https://github.com/williambdean/conjugate", "https://github.com/williambdean/conjugate"), with contribution guidelines at #link("https://github.com/williambdean/conjugate/blob/main/CONTRIBUTING.md", "CONTRIBUTING.md"). Live documentation and an interactive Distribution Explorer are available at #link("https://williambdean.github.io/conjugate/", "https://williambdean.github.io/conjugate/").

#bibliography("paper.bib")
