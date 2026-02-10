---
comments: true
---

# World Cup Problem: From Think Bayes Chapter 18

This example ports the World Cup problem from Allen Downey's [Think Bayes Chapter 18](https://allendowney.github.io/ThinkBayes2/chap18.html#the-world-cup-problem-revisited) to demonstrate the power of conjugate priors for Poisson processes.

The problem: modeling goal-scoring rates in soccer games using a Poisson process, where the gamma distribution serves as the conjugate prior. What takes complex grid computations in the original can be solved with simple parameter updates using conjugate-models.

## Import modules

Import the required distributions and functions:

- `Gamma`: Prior distribution for Poisson rate parameter
- `Poisson`: Likelihood function for goal counts
- `GammaPoisson`: Predictive distribution for Poisson model
- `poisson_gamma`: Posterior update function
- `poisson_gamma_predictive`: Predictive distribution function

```python
from conjugate.distributions import Gamma, Poisson, GammaPoisson
from conjugate.models import poisson_gamma, poisson_gamma_predictive

import matplotlib.pyplot as plt
import numpy as np
```

## Observed Data

From Think Bayes: Germany vs Brazil 2014 World Cup where Brazil scored 4 goals. The data represents the total goals (`x_total`) observed over a time period (`n` games).

```python
# World Cup data: Brazil scored 4 goals in 1 game
x_total = 4
n_games = 1
```

## Bayesian Inference

### Posterior Distribution

Using the Gamma-Poisson conjugate relationship, we can update our prior with the observed data. The beauty of conjugate priors is that the posterior is also a Gamma distribution with updated parameters:

```python
# Prior: Gamma(alpha=1.4, beta=1) from Think Bayes
prior = Gamma(alpha=1.4, beta=1)

# Posterior update using conjugate relationship
posterior = poisson_gamma(x_total=x_total, n=n_games, prior=prior)

print(f"Prior parameters: alpha={prior.alpha}, beta={prior.beta}")
print(f"Posterior parameters: alpha={posterior.alpha}, beta={posterior.beta}")
print(f"Prior mean: {prior.mean():.3f}")
print(f"Posterior mean: {posterior.mean():.3f}")
```

The posterior parameters follow the simple conjugate update rule:
- `α_posterior = α_prior + x_total` (adding observed goals)
- `β_posterior = β_prior + n_games` (adding observed time)

### Predictive Distribution

Get the predictive distribution for future games using the posterior:

```python
# Predictive distribution for next game
posterior_predictive = poisson_gamma_predictive(
    n=1,
    distribution=posterior
)

print(f"Predicted goals in next game: {posterior_predictive.mean():.2f}")
```

## Additional Analysis

Compare prior and posterior distributions to see how the data updates our beliefs about goal-scoring rates:

```python
# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Top plot: Prior vs Posterior distributions
ax1 = axes[0]
lam_range = np.linspace(0, 8, 200)

# Plot prior
prior_pdf = prior.dist.pdf(lam_range)
ax1.plot(lam_range, prior_pdf, label='Prior', color='blue', linewidth=2)

# Plot posterior
posterior_pdf = posterior.dist.pdf(lam_range)
ax1.plot(lam_range, posterior_pdf, label='Posterior', color='red', linewidth=2)

# Add vertical lines for means
ax1.axvline(prior.mean(), color='blue', linestyle='--', alpha=0.7, label=f'Prior mean: {prior.mean():.2f}')
ax1.axvline(posterior.mean(), color='red', linestyle='--', alpha=0.7, label=f'Posterior mean: {posterior.mean():.2f}')
ax1.axvline(x_total/n_games, color='black', linestyle=':', alpha=0.7, label=f'Observed rate: {x_total/n_games:.2f}')

ax1.set_xlabel('Goal scoring rate (λ)')
ax1.set_ylabel('Probability Density')
ax1.set_title('Prior vs Posterior for Goal Scoring Rate')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom plot: Predictive distribution
ax2 = axes[1]
goals_range = np.arange(0, 11)

# Plot prior predictive
prior_predictive = poisson_gamma_predictive(n=1, distribution=prior)
prior_pmf = prior_predictive.dist.pmf(goals_range)
ax2.bar(goals_range - 0.15, prior_pmf, width=0.3, label='Prior Predictive', color='blue', alpha=0.7)

# Plot posterior predictive
post_pmf = posterior_predictive.dist.pmf(goals_range)
ax2.bar(goals_range + 0.15, post_pmf, width=0.3, label='Posterior Predictive', color='red', alpha=0.7)

# Add observed data
ax2.axvline(x_total, color='black', linestyle=':', linewidth=2, label=f'Observed: {x_total} goals')

ax2.set_xlabel('Number of Goals')
ax2.set_ylabel('Probability')
ax2.set_title('Predictive Distribution for Next Game')
ax2.set_xticks(goals_range)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Connection to Think Bayes

### The Conjugate Advantage

In Think Bayes, Allen Downey shows the grid method requiring:
1. Creating a grid of λ values
2. Computing Poisson likelihood for each λ
3. Multiplying prior × likelihood and normalizing
4. Multiple lines of complex code

With conjugate-models, this reduces to:
```python
posterior = poisson_gamma(x_total=4, n=1, prior=prior)
```

### Mathematical Intuition

The Gamma-Poisson conjugate relationship works because both distributions share the same functional form:
- Gamma prior: λ^(α-1) e^(-βλ)
- Poisson likelihood: λ^k e^(-λt)
- Posterior: λ^(α-1+k) e^(-λ(β+t))

This elegant mathematical property gives us the simple update rules:
- `α_posterior = α_prior + observed_goals`
- `β_posterior = β_prior + observed_time`

### What the Parameters Mean

- `α` (alpha): Represents "pseudo-goals" - prior knowledge about goal counts
- `β` (beta): Represents "pseudo-games" - prior knowledge about time periods
- The ratio `α/β` gives the expected goal-scoring rate

This example perfectly demonstrates why conjugate priors are so powerful - they turn complex Bayesian updates into simple arithmetic operations while maintaining full probabilistic inference.

---

*This example is inspired by Allen Downey's Think Bayes, Chapter 18: [Conjugate Priors](https://allendowney.github.io/ThinkBayes2/chap18.html)*
