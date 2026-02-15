---
hide:
    - navigation
comments: true
---
# Quick Reference

A cheat sheet for conjugate model pairs to help you quickly find the right functions for your use case.

## How to Use This Table

1. Find your **likelihood** (the distribution of your data)
2. Look up the corresponding **model function** from `conjugate.models`
3. Check if there's a **helper function** to extract sufficient statistics from raw data
4. See **examples** for practical usage patterns

---

## Discrete Likelihoods

| Likelihood | Prior / Posterior | Model | Helper | Examples |
|------------|-------------------|-------|--------|----------|
| Binomial | [Beta] | [`binomial_beta`][binomial_beta] | [`binomial_beta_inputs`][binomial_beta_inputs] | [Binomial], [Bandit], [Bayesian Update], [Vectorized] |
| Bernoulli | [Beta] | [`bernoulli_beta`][bernoulli_beta] | [`bernoulli_beta_inputs`][bernoulli_beta_inputs] | [Raw Data Workflow] |
| Negative Binomial | [Beta] | [`negative_binomial_beta`][negative_binomial_beta] | [`negative_binomial_beta_inputs`][negative_binomial_beta_inputs] | - |
| Geometric | [Beta] | [`geometric_beta`][geometric_beta] | [`geometric_beta_inputs`][geometric_beta_inputs] | - |
| Hypergeometric | [BetaBinomial] | [`hypergeometric_beta_binomial`][hypergeometric_beta_binomial] | [`hypergeometric_beta_binomial_inputs`][hypergeometric_beta_binomial_inputs] | - |
| Categorical | [Dirichlet] | [`categorical_dirichlet`][categorical_dirichlet] | [`categorical_dirichlet_inputs`][categorical_dirichlet_inputs] | - |
| Multinomial | [Dirichlet] | [`multinomial_dirichlet`][multinomial_dirichlet] | [`multinomial_dirichlet_inputs`][multinomial_dirichlet_inputs] | [Raw Data Workflow] |
| Poisson | [Gamma] | [`poisson_gamma`][poisson_gamma] | [`poisson_gamma_inputs`][poisson_gamma_inputs] | [Bootstrap], [Limit], [Raw Data Workflow] |

## Continuous Likelihoods

| Likelihood | Prior / Posterior | Model | Helper | Examples |
|------------|-------------------|-------|--------|----------|
| Exponential | [Gamma] | [`exponential_gamma`][exponential_gamma] | - | [Thompson], [Raw Data Workflow] |
| Gamma (known shape) | [Gamma] | [`gamma_known_shape`][gamma_known_shape] | [`gamma_known_shape_inputs`][gamma_known_shape_inputs] | - |
| Gamma (known rate) | [GammaKnownRateProportional] | [`gamma_known_rate`][gamma_known_rate] | [`gamma_known_rate_inputs`][gamma_known_rate_inputs] | - |
| Gamma | [GammaProportional] | [`gamma`][gamma] | [`gamma_inputs`][gamma_inputs] | - |
| Inverse Gamma (known rate) | [Gamma] | [`inverse_gamma_known_rate`][inverse_gamma_known_rate] | [`inverse_gamma_known_rate_inputs`][inverse_gamma_known_rate_inputs] | - |
| Normal (known variance) | [Normal] | [`normal_known_variance`][normal_known_variance] | [`normal_known_variance_inputs`][normal_known_variance_inputs] | [Shortest Path] |
| Normal (known precision) | [Normal] | [`normal_known_precision`][normal_known_precision] | [`normal_known_precision_inputs`][normal_known_precision_inputs] | - |
| Normal (known mean) | [InverseGamma] | [`normal_known_mean`][normal_known_mean] | [`normal_known_mean_inputs`][normal_known_mean_inputs] | - |
| Normal | [NormalInverseGamma] | [`normal`][normal] | [`normal_inputs`][normal_inputs] | - |
| Linear Regression | [NormalInverseGamma] | [`linear_regression`][linear_regression] | [`linear_regression_inputs`][linear_regression_inputs] | [Linear Regression] |
| Log Normal | [NormalInverseGamma] | [`log_normal`][log_normal] | [`log_normal_inputs`][log_normal_inputs] | - |
| Uniform | [Pareto] | [`uniform_pareto`][uniform_pareto] | [`uniform_pareto_inputs`][uniform_pareto_inputs] | - |
| Pareto | [Gamma] | [`pareto_gamma`][pareto_gamma] | [`pareto_gamma_inputs`][pareto_gamma_inputs] | [Unsupported Distributions] |
| Beta | [BetaProportional] | [`beta`][beta] | [`beta_inputs`][beta_inputs] | [Raw Data Workflow] |
| Von Mises (known κ) | [VonMisesKnownConcentration] | [`von_mises_known_concentration`][von_mises_known_concentration] | [`von_mises_known_concentration_inputs`][von_mises_known_concentration_inputs] | - |
| Von Mises (known μ) | [VonMisesKnownDirectionProportional] | [`von_mises_known_direction`][von_mises_known_direction] | [`von_mises_known_direction_inputs`][von_mises_known_direction_inputs] | - |
| Weibull (known shape) | [InverseGamma] | [`weibull_inverse_gamma_known_shape`][weibull_inverse_gamma_known_shape] | [`weibull_inverse_gamma_known_shape_inputs`][weibull_inverse_gamma_known_shape_inputs] | - |

## Multivariate

| Likelihood | Prior / Posterior | Model | Helper | Examples |
|------------|-------------------|-------|--------|----------|
| Multivariate Normal (known cov) | [MultivariateNormal] | [`multivariate_normal_known_covariance`][multivariate_normal_known_covariance] | [`multivariate_normal_known_covariance_inputs`][multivariate_normal_known_covariance_inputs] | - |
| Multivariate Normal (known precision) | [MultivariateNormal] | [`multivariate_normal_known_precision`][multivariate_normal_known_precision] | [`multivariate_normal_known_precision_inputs`][multivariate_normal_known_precision_inputs] | - |
| Multivariate Normal (known mean) | [InverseWishart] | [`multivariate_normal_known_mean`][multivariate_normal_known_mean] | [`multivariate_normal_known_mean_inputs`][multivariate_normal_known_mean_inputs] | - |
| Multivariate Normal | [NormalInverseWishart] | [`multivariate_normal`][multivariate_normal] | [`multivariate_normal_inputs`][multivariate_normal_inputs] | - |

---

## See Also

- [Models API Reference](models.md) - Full documentation for all model functions
- [Distributions API Reference](distributions.md) - Full documentation for all distribution classes
- [Helpers API Reference](helpers.md) - Full documentation for all helper functions
- [Raw Data Workflow](examples/raw-data-workflow.md) - Complete examples from raw data to posterior

<!-- Distribution Links -->
[Beta]: distributions.md#conjugate.distributions.Beta
[BetaBinomial]: distributions.md#conjugate.distributions.BetaBinomial
[BetaProportional]: distributions.md#conjugate.distributions.BetaProportional
[Dirichlet]: distributions.md#conjugate.distributions.Dirichlet
[Gamma]: distributions.md#conjugate.distributions.Gamma
[GammaKnownRateProportional]: distributions.md#conjugate.distributions.GammaKnownRateProportional
[GammaProportional]: distributions.md#conjugate.distributions.GammaProportional
[InverseGamma]: distributions.md#conjugate.distributions.InverseGamma
[InverseWishart]: distributions.md#conjugate.distributions.InverseWishart
[MultivariateNormal]: distributions.md#conjugate.distributions.MultivariateNormal
[Normal]: distributions.md#conjugate.distributions.Normal
[NormalInverseGamma]: distributions.md#conjugate.distributions.NormalInverseGamma
[NormalInverseWishart]: distributions.md#conjugate.distributions.NormalInverseWishart
[Pareto]: distributions.md#conjugate.distributions.Pareto
[VonMisesKnownConcentration]: distributions.md#conjugate.distributions.VonMisesKnownConcentration
[VonMisesKnownDirectionProportional]: distributions.md#conjugate.distributions.VonMisesKnownDirectionProportional

<!-- Model Links -->
[binomial_beta]: models.md#conjugate.models.binomial_beta
[bernoulli_beta]: models.md#conjugate.models.bernoulli_beta
[negative_binomial_beta]: models.md#conjugate.models.negative_binomial_beta
[geometric_beta]: models.md#conjugate.models.geometric_beta
[hypergeometric_beta_binomial]: models.md#conjugate.models.hypergeometric_beta_binomial
[categorical_dirichlet]: models.md#conjugate.models.categorical_dirichlet
[multinomial_dirichlet]: models.md#conjugate.models.multinomial_dirichlet
[poisson_gamma]: models.md#conjugate.models.poisson_gamma
[exponential_gamma]: models.md#conjugate.models.exponential_gamma
[gamma_known_shape]: models.md#conjugate.models.gamma_known_shape
[gamma_known_rate]: models.md#conjugate.models.gamma_known_rate
[gamma]: models.md#conjugate.models.gamma
[inverse_gamma_known_rate]: models.md#conjugate.models.inverse_gamma_known_rate
[normal_known_variance]: models.md#conjugate.models.normal_known_variance
[normal_known_precision]: models.md#conjugate.models.normal_known_precision
[normal_known_mean]: models.md#conjugate.models.normal_known_mean
[normal]: models.md#conjugate.models.normal
[linear_regression]: models.md#conjugate.models.linear_regression
[log_normal]: models.md#conjugate.models.log_normal
[uniform_pareto]: models.md#conjugate.models.uniform_pareto
[pareto_gamma]: models.md#conjugate.models.pareto_gamma
[beta]: models.md#conjugate.models.beta
[von_mises_known_concentration]: models.md#conjugate.models.von_mises_known_concentration
[von_mises_known_direction]: models.md#conjugate.models.von_mises_known_direction
[weibull_inverse_gamma_known_shape]: models.md#conjugate.models.weibull_inverse_gamma_known_shape
[multivariate_normal_known_covariance]: models.md#conjugate.models.multivariate_normal_known_covariance
[multivariate_normal_known_precision]: models.md#conjugate.models.multivariate_normal_known_precision
[multivariate_normal_known_mean]: models.md#conjugate.models.multivariate_normal_known_mean
[multivariate_normal]: models.md#conjugate.models.multivariate_normal

<!-- Helper Links -->
[binomial_beta_inputs]: helpers.md#conjugate.helpers.binomial_beta_inputs
[bernoulli_beta_inputs]: helpers.md#conjugate.helpers.bernoulli_beta_inputs
[negative_binomial_beta_inputs]: helpers.md#conjugate.helpers.negative_binomial_beta_inputs
[geometric_beta_inputs]: helpers.md#conjugate.helpers.geometric_beta_inputs
[hypergeometric_beta_binomial_inputs]: helpers.md#conjugate.helpers.hypergeometric_beta_binomial_inputs
[categorical_dirichlet_inputs]: helpers.md#conjugate.helpers.categorical_dirichlet_inputs
[multinomial_dirichlet_inputs]: helpers.md#conjugate.helpers.multinomial_dirichlet_inputs
[poisson_gamma_inputs]: helpers.md#conjugate.helpers.poisson_gamma_inputs
[exponential_gamma_inputs]: helpers.md#conjugate.helpers.exponential_gamma_inputs
[gamma_known_shape_inputs]: helpers.md#conjugate.helpers.gamma_known_shape_inputs
[gamma_known_rate_inputs]: helpers.md#conjugate.helpers.gamma_known_rate_inputs
[gamma_inputs]: helpers.md#conjugate.helpers.gamma_inputs
[inverse_gamma_known_rate_inputs]: helpers.md#conjugate.helpers.inverse_gamma_known_rate_inputs
[normal_known_variance_inputs]: helpers.md#conjugate.helpers.normal_known_variance_inputs
[normal_known_precision_inputs]: helpers.md#conjugate.helpers.normal_known_precision_inputs
[normal_known_mean_inputs]: helpers.md#conjugate.helpers.normal_known_mean_inputs
[normal_inputs]: helpers.md#conjugate.helpers.normal_inputs
[linear_regression_inputs]: helpers.md#conjugate.helpers.linear_regression_inputs
[log_normal_inputs]: helpers.md#conjugate.helpers.log_normal_inputs
[uniform_pareto_inputs]: helpers.md#conjugate.helpers.uniform_pareto_inputs
[pareto_gamma_inputs]: helpers.md#conjugate.helpers.pareto_gamma_inputs
[beta_inputs]: helpers.md#conjugate.helpers.beta_inputs
[von_mises_known_concentration_inputs]: helpers.md#conjugate.helpers.von_mises_known_concentration_inputs
[von_mises_known_direction_inputs]: helpers.md#conjugate.helpers.von_mises_known_direction_inputs
[weibull_inverse_gamma_known_shape_inputs]: helpers.md#conjugate.helpers.weibull_inverse_gamma_known_shape_inputs
[multivariate_normal_known_covariance_inputs]: helpers.md#conjugate.helpers.multivariate_normal_known_covariance_inputs
[multivariate_normal_known_precision_inputs]: helpers.md#conjugate.helpers.multivariate_normal_known_precision_inputs
[multivariate_normal_known_mean_inputs]: helpers.md#conjugate.helpers.multivariate_normal_known_mean_inputs
[multivariate_normal_inputs]: helpers.md#conjugate.helpers.multivariate_normal_inputs

<!-- Example Links -->
[Binomial]: examples/binomial.md
[Bandit]: examples/bandit.md
[Bayesian Update]: examples/bayesian-update.md
[Vectorized]: examples/vectorized-inputs.md
[Raw Data Workflow]: examples/raw-data-workflow.md
[Bootstrap]: examples/bootstrap.md
[Limit]: examples/limit.md
[Thompson]: examples/thompson.md
[Shortest Path]: examples/shortest-path-itt.md
[Linear Regression]: examples/linear-regression.md
[Unsupported Distributions]: examples/unsupported-distributions.md
