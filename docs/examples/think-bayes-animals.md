---
comments: true
---

# Lions, Tigers and Bears: From Think Bayes Chapter 18

This example ports the Lions, Tigers and Bears problem from Allen Downey's [Think Bayes Chapter 18](https://allendowney.github.io/ThinkBayes2/chap18.html#lions-and-tigers-and-bears) to demonstrate the Dirichlet-Multinomial conjugate relationship.

The problem: estimating the prevalence of different animal species in a wildlife preserve, given observed counts. The Dirichlet distribution serves as the conjugate prior for the Multinomial likelihood, extending the Beta-Binomial concept to multiple categories.

## Import modules

Import the required distributions and functions:

- `Dirichlet`: Prior and posterior distribution for species probabilities
- `Multinomial`: Likelihood function for species counts
- `DirichletMultinomial`: Predictive distribution for multinomial model
- `multinomial_dirichlet`: Posterior update function
- `multinomial_dirichlet_predictive`: Predictive distribution function

```python
from conjugate.distributions import Dirichlet, Multinomial, DirichletMultinomial
from conjugate.models import multinomial_dirichlet, multinomial_dirichlet_predictive

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

## Observed Data

From Think Bayes: during a tour we observed 3 lions, 2 tigers, and 1 bear. We want to estimate the prevalence (probability) of each species.

```python
# Animal observation data
observed_counts = np.array([3, 2, 1])  # [lions, tigers, bears]
species_names = ['Lion', 'Tiger', 'Bear']
total_observed = observed_counts.sum()

print(f"Observed counts: {dict(zip(species_names, observed_counts))}")
print(f"Total animals observed: {total_observed}")
```

## Bayesian Inference

### Posterior Distribution

Using the Dirichlet-Multinomial conjugate relationship, we update our prior with observed counts. The posterior is also a Dirichlet distribution:

```python
# Prior: Uniform Dirichlet(alpha=[1, 1, 1]) from Think Bayes
prior = Dirichlet(alpha=np.array([1, 1, 1]))

# Posterior update using conjugate relationship
posterior = multinomial_dirichlet(x=observed_counts, prior=prior)

print(f"Prior parameters: alpha={prior.alpha}")
print(f"Posterior parameters: alpha={posterior.alpha}")
print(f"Prior mean probabilities: {prior.mean()}")
print(f"Posterior mean probabilities: {posterior.mean()}")

# Display as DataFrame for clarity
results_df = pd.DataFrame({
    'Species': species_names,
    'Observed': observed_counts,
    'Prior Mean': prior.mean(),
    'Posterior Mean': posterior.mean(),
    'Observed Proportion': observed_counts / total_observed
})
print(results_df.round(3))
```

The posterior follows the simple conjugate update rule:
`α_posterior = α_prior + observed_counts`

### Predictive Distribution

Get the predictive distribution for future observations:

```python
# Predictive distribution for next 10 animals
n_future = 10
posterior_predictive = multinomial_dirichlet_predictive(
    n=n_future,
    distribution=posterior
)

print(f"Expected composition of next {n_future} animals:")
expected_composition = posterior_predictive.mean()
print(dict(zip(species_names, expected_composition.round(2))))
```

## Additional Analysis

### Probability of Next Animal Being a Bear

The key question from Think Bayes: what's the probability the next animal is a bear?

```python
# Probability next animal is a bear (marginal from posterior)
bear_prob = posterior.mean()[2]  # Third component corresponds to bears
print(f"Probability next animal is a bear: {bear_prob:.3f}")

# Probability distribution for next animal
next_animal_probs = posterior.mean()
print("Probability distribution for next animal:")
for species, prob in zip(species_names, next_animal_probs):
    print(f"  {species}: {prob:.3f}")
```

### Visualization

Create comprehensive visualizations of the analysis:

```python
# Create comprehensive visualization
fig = plt.figure(figsize=(15, 10))

# 1. Prior vs Posterior mean probabilities
ax1 = plt.subplot(2, 3, 1)
x_pos = np.arange(len(species_names))
width = 0.35

ax1.bar(x_pos - width/2, prior.mean(), width, label='Prior Mean', color='blue', alpha=0.7)
ax1.bar(x_pos + width/2, posterior.mean(), width, label='Posterior Mean', color='red', alpha=0.7)
ax1.bar(x_pos + width*1.5, observed_counts/total_observed, width, label='Observed', color='green', alpha=0.7)

ax1.set_xlabel('Species')
ax1.set_ylabel('Probability')
ax1.set_title('Prior vs Posterior Species Probabilities')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(species_names)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Dirichlet distributions (marginals)
ax2 = plt.subplot(2, 3, 2)
p_range = np.linspace(0, 1, 200)

colors = ['blue', 'orange', 'green']
for i, (species, color) in enumerate(zip(species_names, colors)):
    # Get marginal beta distribution for this species
    alpha_i = posterior.alpha[i]
    alpha_sum = posterior.alpha.sum()

    # Marginal beta distribution
    from scipy.stats import beta as beta_dist
    marginal_beta = beta_dist(alpha_i, alpha_sum - alpha_i)
    pdf = marginal_beta.pdf(p_range)

    ax2.plot(p_range, pdf, label=f'{species}', color=color, linewidth=2)
    ax2.axvline(posterior.mean()[i], color=color, linestyle='--', alpha=0.7)

ax2.set_xlabel('Probability')
ax2.set_ylabel('Density')
ax2.set_title('Marginal Posterior Distributions')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Predictive distribution for next 10 animals
ax3 = plt.subplot(2, 3, 3)

# Sample from posterior predictive distribution
n_samples = 1000
predictive_samples = posterior_predictive.dist.rvs(size=n_samples, random_state=42)

# Plot histogram of bear counts in samples
bear_counts = predictive_samples[:, 2]  # Bear counts
ax3.hist(bear_counts, bins=range(0, n_future+2), alpha=0.7, color='green', edgecolor='black')
ax3.axvline(bear_counts.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {bear_counts.mean():.1f}')
ax3.set_xlabel(f'Number of Bears in {n_future} Animals')
ax3.set_ylabel('Frequency')
ax3.set_title('Predictive Distribution: Future Observations')
ax3.set_xticks(range(0, n_future+1))
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Prior vs Posterior parameter comparison
ax4 = plt.subplot(2, 3, 4)

ax4.scatter(prior.alpha, species_names, s=100, label='Prior α', color='blue', alpha=0.7)
ax4.scatter(posterior.alpha, species_names, s=100, label='Posterior α', color='red', alpha=0.7)

# Add lines to show updates
for i, species in enumerate(species_names):
    ax4.plot([prior.alpha[i], posterior.alpha[i]], [i, i], 'k--', alpha=0.3)
    ax4.text((prior.alpha[i] + posterior.alpha[i])/2, i + 0.1,
             f'+{observed_counts[i]}', ha='center', fontsize=9)

ax4.set_xlabel('Alpha Parameter')
ax4.set_ylabel('Species')
ax4.set_title('Parameter Update: Prior → Posterior')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Correlation matrix (posterior uncertainty)
ax5 = plt.subplot(2, 3, 5)

# Calculate covariance matrix from posterior samples
posterior_samples = posterior.dist.rvs(size=n_samples, random_state=42)
cov_matrix = np.cov(posterior_samples.T)

im = ax5.imshow(cov_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.05, vmax=0.05)
ax5.set_xticks(range(len(species_names)))
ax5.set_yticks(range(len(species_names)))
ax5.set_xticklabels(species_names)
ax5.set_yticklabels(species_names)
ax5.set_title('Posterior Covariance Matrix')

# Add correlation values
for i in range(len(species_names)):
    for j in range(len(species_names)):
        corr = cov_matrix[i,j] / np.sqrt(cov_matrix[i,i] * cov_matrix[j,j])
        ax5.text(j, i, f'{corr:.2f}', ha='center', va='center',
                fontdict={'color': 'white' if abs(corr) > 0.02 else 'black'})

plt.colorbar(im, ax=ax5, shrink=0.8)

# 6. Posterior uncertainty visualization
ax6 = plt.subplot(2, 3, 6)

# Create box plots of posterior samples
posterior_samples_df = pd.DataFrame(posterior_samples, columns=species_names)
posterior_samples_df.boxplot(ax=ax6, patch_artist=True)

ax6.set_ylabel('Probability')
ax6.set_title('Posterior Uncertainty (Box Plots)')
ax6.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

## Connection to Think Bayes

### The Multivariate Conjugate Advantage

The Dirichlet-Multinomial conjugate extends the Beta-Binomial concept to multiple categories:

- **Univariate**: Beta prior for single probability `p` with Binomial likelihood
- **Multivariate**: Dirichlet prior for probability vector `p` with Multinomial likelihood

In Think Bayes, this would require:
1. Creating a 3D grid of probability combinations
2. Ensuring probabilities sum to 1 at each grid point
3. Computing multinomial likelihood everywhere
4. Complex normalization

With conjugate-models:
```python
posterior = multinomial_dirichlet(x=observed_counts, prior=prior)
```

### Mathematical Intuition

The Dirichlet-Multinomial conjugate works because both involve products of probabilities raised to powers:

- Dirichlet prior: ∏ p_i^(α_i-1)
- Multinomial likelihood: ∏ p_i^x_i
- Posterior: ∏ p_i^(α_i-1+x_i)

This gives the simple vector update:
`α_posterior = α_prior + observed_counts`

### What the Parameters Mean

- `α_i` (alpha_i): "pseudo-counts" for species i - prior beliefs about each species
- Sum `Σα_i`: Represents the strength of prior belief (equivalent sample size)
- The ratio `α_i/(Σα_i)` gives the expected probability for species i

### Key Insights

1. **Bear Probability**: The probability the next animal is a bear is simply the posterior mean for bears: `(1+1)/(3+6) = 2/9 ≈ 22.2%`

2. **Correlation Structure**: The Dirichlet induces negative correlations between species probabilities (since they must sum to 1)

3. **Learning Behavior**: With more data, the posterior becomes more concentrated around the observed proportions

This example beautifully demonstrates how conjugate priors extend naturally to multivariate problems, turning potentially complex calculations into simple vector arithmetic.

---

*This example is inspired by Allen Downey's Think Bayes, Chapter 18: [Conjugate Priors](https://allendowney.github.io/ThinkBayes2/chap18.html)*
