---
comments: true
---

# Euro Problem: From Think Bayes Chapter 18

This example ports the Euro problem from Allen Downey's [Think Bayes Chapter 18](https://allendowney.github.io/ThinkBayes2/chap18.html#binomial-likelihood) to demonstrate the Beta-Binomial conjugate relationship.

The problem: estimating the probability of a coin landing heads up, given 140 heads out of 250 flips. The Beta distribution serves as the conjugate prior for the Binomial likelihood, making Bayesian updates trivial.

## Import modules

Import the required distributions and functions:

- `Beta`: Prior and posterior distribution for success probability
- `Binomial`: Likelihood function for coin flips
- `BetaBinomial`: Predictive distribution for binomial model
- `binomial_beta`: Posterior update function
- `binomial_beta_predictive`: Predictive distribution function

```python
from conjugate.distributions import Beta, Binomial, BetaBinomial
from conjugate.models import binomial_beta, binomial_beta_predictive

import matplotlib.pyplot as plt
import numpy as np
```

## Observed Data

From Think Bayes: 140 heads out of 250 coin flips. We're estimating the probability `p` of getting heads.

```python
# Euro coin data: 140 heads out of 250 flips
x = 140  # number of successes (heads)
n = 250   # number of trials (flips)
```

## Bayesian Inference

### Posterior Distribution

Using the Beta-Binomial conjugate relationship, we update our prior with the observed data. The posterior is also a Beta distribution with updated parameters:

```python
# Prior: Uniform distribution Beta(alpha=1, beta=1) from Think Bayes
prior = Beta(alpha=1, beta=1)

# Posterior update using conjugate relationship
posterior = binomial_beta(n=n, x=x, prior=prior)

print(f"Prior parameters: alpha={prior.alpha}, beta={prior.beta}")
print(f"Posterior parameters: alpha={posterior.alpha}, beta={posterior.beta}")
print(f"Prior mean: {prior.mean():.3f}")
print(f"Posterior mean: {posterior.mean():.3f}")
print(f"Observed proportion: {x/n:.3f}")
```

The posterior follows the simple conjugate update rule:
- `α_posterior = α_prior + x` (adding observed heads)
- `β_posterior = β_prior + (n - x)` (adding observed tails)

### Predictive Distribution

Get the predictive distribution for future coin flips:

```python
# Predictive distribution for next 10 flips
n_future = 10
posterior_predictive = binomial_beta_predictive(
    n=n_future,
    distribution=posterior
)

print(f"Predicted heads in {n_future} flips: {posterior_predictive.mean():.2f}")
```

## Additional Analysis

Compare prior, posterior, and predictive distributions:

```python
# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Top plot: Prior vs Posterior distributions
ax1 = axes[0]
p_range = np.linspace(0, 1, 200)

# Plot prior (uniform)
prior_pdf = prior.dist.pdf(p_range)
ax1.plot(p_range, prior_pdf, label='Prior (Uniform)', color='blue', linewidth=2)

# Plot posterior
posterior_pdf = posterior.dist.pdf(p_range)
ax1.plot(p_range, posterior_pdf, label='Posterior', color='red', linewidth=2)

# Add vertical lines
ax1.axvline(prior.mean(), color='blue', linestyle='--', alpha=0.7, label=f'Prior mean: {prior.mean():.3f}')
ax1.axvline(posterior.mean(), color='red', linestyle='--', alpha=0.7, label=f'Posterior mean: {posterior.mean():.3f}')
ax1.axvline(x/n, color='black', linestyle=':', alpha=0.7, label=f'Observed: {x/n:.3f}')

ax1.set_xlabel('Probability of Heads (p)')
ax1.set_ylabel('Probability Density')
ax1.set_title('Prior vs Posterior for Coin Bias')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom plot: Predictive distributions
ax2 = axes[1]
heads_range = np.arange(0, n_future + 1)

# Plot prior predictive
prior_predictive = binomial_beta_predictive(n=n_future, distribution=prior)
prior_pmf = prior_predictive.dist.pmf(heads_range)
ax2.bar(heads_range - 0.2, prior_pmf, width=0.4, label='Prior Predictive', color='blue', alpha=0.7)

# Plot posterior predictive
post_pmf = posterior_predictive.dist.pmf(heads_range)
ax2.bar(heads_range + 0.2, post_pmf, width=0.4, label='Posterior Predictive', color='red', alpha=0.7)

# Add expected value
ax2.axvline(x * n_future / n, color='black', linestyle=':', linewidth=2,
            label=f'Expected: {x * n_future / n:.1f} heads')

ax2.set_xlabel(f'Number of Heads in {n_future} Future Flips')
ax2.set_ylabel('Probability')
ax2.set_title('Predictive Distribution for Future Flips')
ax2.set_xticks(heads_range)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Credible Interval

Calculate a 95% credible interval for the coin bias:

```python
# 95% credible interval
ci_lower = posterior.dist.ppf(0.025)
ci_upper = posterior.dist.ppf(0.975)

print(f"95% Credible Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"Interval width: {ci_upper - ci_lower:.3f}")
```

## Connection to Think Bayes

### The Conjugate Advantage

In Think Bayes, the grid method requires:
1. Creating a grid of probability values
2. Computing binomial likelihood for each probability
3. Multiplying prior × likelihood and normalizing
4. Complex visualization code

With conjugate-models:
```python
posterior = binomial_beta(n=250, x=140, prior=prior)
```

### Mathematical Intuition

The Beta-Binomial conjugate works because both share the same functional form:
- Beta prior: p^(α-1) (1-p)^(β-1)
- Binomial likelihood: p^k (1-p)^(n-k)
- Posterior: p^(α-1+k) (1-p)^(β-1+n-k)

This gives the simple update rules:
- `α_posterior = α_prior + successes`
- `β_posterior = β_prior + failures`

### What the Parameters Mean

- `α` (alpha): "pseudo-successes" - prior beliefs about heads
- `β` (beta): "pseudo-failures" - prior beliefs about tails
- The ratio `α/(α+β)` gives the expected probability of heads
- The sum `α+β` represents the strength of prior belief (equivalent sample size)

### Comparison with Grid Method

The grid method from Think Bayes requires discretizing the probability space and computing likelihoods at each point. The conjugate approach:

1. **Computationally Efficient**: O(1) vs O(grid_size) operations
2. **Exact Solution**: No discretization error
3. **Analytical Properties**: Easy to compute moments, credible intervals
4. **Scalable**: Works equally well for any data size

This example showcases the elegance and efficiency of conjugate Bayesian analysis compared to grid approximation methods.

---

*This example is inspired by Allen Downey's Think Bayes, Chapter 18: [Conjugate Priors](https://allendowney.github.io/ThinkBayes2/chap18.html)*
