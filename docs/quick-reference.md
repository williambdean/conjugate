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
3. Check the **Functions** column for:
   - **Helper** functions to extract sufficient statistics from raw data
   - **Predictive** functions for posterior predictive distributions
4. See **examples** for practical usage patterns

**Note:** Not all models have closed-form predictive distributions. When available, predictive functions follow the pattern `<model>_predictive`.

---

## Coverage

This reference includes **26 conjugate model pairs** with:

- ✅ **Model functions** for all 26 pairs (posterior updates)
- ✅ **Helper functions** for 25 pairs (raw data → sufficient statistics)
- ✅ **Predictive functions** for 18 pairs (posterior predictive distributions)

Predictive functions are available for the most commonly used models, particularly those based on Beta, Gamma, and Normal priors.

**Note:** For models without closed-form predictive distributions, see [Unsupported Distributions] for a guide on generating posterior predictive samples using sampling techniques.

---

## Discrete Likelihoods

| Likelihood | Prior / Posterior | Model | Functions | Examples |
|------------|-------------------|-------|-----------|----------|
| Binomial | [Beta] | [`binomial_beta`][binomial_beta] | Helper: [`binomial_beta_inputs`][binomial_beta_inputs]<br>Predictive: [`binomial_beta_predictive`][binomial_beta_predictive] | [Binomial], [Bandit], [Bayesian Update], [Vectorized] |
| Bernoulli | [Beta] | [`bernoulli_beta`][bernoulli_beta] | Helper: [`bernoulli_beta_inputs`][bernoulli_beta_inputs]<br>Predictive: [`bernoulli_beta_predictive`][bernoulli_beta_predictive] | [Raw Data Workflow] |
| Negative Binomial | [Beta] | [`negative_binomial_beta`][negative_binomial_beta] | Helper: [`negative_binomial_beta_inputs`][negative_binomial_beta_inputs]<br>Predictive: [`negative_binomial_beta_predictive`][negative_binomial_beta_predictive] | - |
| Geometric | [Beta] | [`geometric_beta`][geometric_beta] | Helper: [`geometric_beta_inputs`][geometric_beta_inputs]<br>Predictive: [`geometric_beta_predictive`][geometric_beta_predictive] | - |
| Hypergeometric | [BetaBinomial] | [`hypergeometric_beta_binomial`][hypergeometric_beta_binomial] | Helper: [`hypergeometric_beta_binomial_inputs`][hypergeometric_beta_binomial_inputs]<br>Predictive: - | - |
| Categorical | [Dirichlet] | [`categorical_dirichlet`][categorical_dirichlet] | Helper: [`categorical_dirichlet_inputs`][categorical_dirichlet_inputs]<br>Predictive: [`categorical_dirichlet_predictive`][categorical_dirichlet_predictive] | - |
| Multinomial | [Dirichlet] | [`multinomial_dirichlet`][multinomial_dirichlet] | Helper: [`multinomial_dirichlet_inputs`][multinomial_dirichlet_inputs]<br>Predictive: [`multinomial_dirichlet_predictive`][multinomial_dirichlet_predictive] | [Raw Data Workflow] |
| Poisson | [Gamma] | [`poisson_gamma`][poisson_gamma] | Helper: [`poisson_gamma_inputs`][poisson_gamma_inputs]<br>Predictive: [`poisson_gamma_predictive`][poisson_gamma_predictive] | [Bootstrap], [Limit], [Raw Data Workflow] |

## Continuous Likelihoods

| Likelihood | Prior / Posterior | Model | Functions | Examples |
|------------|-------------------|-------|-----------|----------|
| Exponential | [Gamma] | [`exponential_gamma`][exponential_gamma] | Helper: -<br>Predictive: [`exponential_gamma_predictive`][exponential_gamma_predictive] | [Thompson], [Raw Data Workflow] |
| Gamma (known shape) | [Gamma] | [`gamma_known_shape`][gamma_known_shape] | Helper: [`gamma_known_shape_inputs`][gamma_known_shape_inputs]<br>Predictive: [`gamma_known_shape_predictive`][gamma_known_shape_predictive] | - |
| Gamma (known rate) | [GammaKnownRateProportional] | [`gamma_known_rate`][gamma_known_rate] | Helper: [`gamma_known_rate_inputs`][gamma_known_rate_inputs]<br>Predictive: - | - |
| Gamma | [GammaProportional] | [`gamma`][gamma] | Helper: [`gamma_inputs`][gamma_inputs]<br>Predictive: - | - |
| Inverse Gamma (known rate) | [Gamma] | [`inverse_gamma_known_rate`][inverse_gamma_known_rate] | Helper: [`inverse_gamma_known_rate_inputs`][inverse_gamma_known_rate_inputs]<br>Predictive: - | - |
| Normal (known variance) | [Normal] | [`normal_known_variance`][normal_known_variance] | Helper: [`normal_known_variance_inputs`][normal_known_variance_inputs]<br>Predictive: [`normal_known_variance_predictive`][normal_known_variance_predictive] | [Shortest Path] |
| Normal (known precision) | [Normal] | [`normal_known_precision`][normal_known_precision] | Helper: [`normal_known_precision_inputs`][normal_known_precision_inputs]<br>Predictive: [`normal_known_precision_predictive`][normal_known_precision_predictive] | - |
| Normal (known mean) | [InverseGamma] | [`normal_known_mean`][normal_known_mean] | Helper: [`normal_known_mean_inputs`][normal_known_mean_inputs]<br>Predictive: [`normal_known_mean_predictive`][normal_known_mean_predictive] | - |
| Normal | [NormalInverseGamma] | [`normal`][normal] | Helper: [`normal_inputs`][normal_inputs]<br>Predictive: [`normal_predictive`][normal_predictive] | - |
| Linear Regression | [NormalInverseGamma] | [`linear_regression`][linear_regression] | Helper: [`linear_regression_inputs`][linear_regression_inputs]<br>Predictive: [`linear_regression_predictive`][linear_regression_predictive] | [Linear Regression] |
| Log Normal | [NormalInverseGamma] | [`log_normal`][log_normal] | Helper: [`log_normal_inputs`][log_normal_inputs]<br>Predictive: - | - |
| Uniform | [Pareto] | [`uniform_pareto`][uniform_pareto] | Helper: [`uniform_pareto_inputs`][uniform_pareto_inputs]<br>Predictive: - | - |
| Pareto | [Gamma] | [`pareto_gamma`][pareto_gamma] | Helper: [`pareto_gamma_inputs`][pareto_gamma_inputs]<br>Predictive: - | [Unsupported Distributions] |
| Beta | [BetaProportional] | [`beta`][beta] | Helper: [`beta_inputs`][beta_inputs]<br>Predictive: - | [Raw Data Workflow] |
| Von Mises (known κ) | [VonMisesKnownConcentration] | [`von_mises_known_concentration`][von_mises_known_concentration] | Helper: [`von_mises_known_concentration_inputs`][von_mises_known_concentration_inputs]<br>Predictive: - | - |
| Von Mises (known μ) | [VonMisesKnownDirectionProportional] | [`von_mises_known_direction`][von_mises_known_direction] | Helper: [`von_mises_known_direction_inputs`][von_mises_known_direction_inputs]<br>Predictive: - | - |
| Weibull (known shape) | [InverseGamma] | [`weibull_inverse_gamma_known_shape`][weibull_inverse_gamma_known_shape] | Helper: [`weibull_inverse_gamma_known_shape_inputs`][weibull_inverse_gamma_known_shape_inputs]<br>Predictive: - | - |

## Multivariate

| Likelihood | Prior / Posterior | Model | Functions | Examples |
|------------|-------------------|-------|-----------|----------|
| Multivariate Normal (known cov) | [MultivariateNormal] | [`multivariate_normal_known_covariance`][multivariate_normal_known_covariance] | Helper: [`multivariate_normal_known_covariance_inputs`][multivariate_normal_known_covariance_inputs]<br>Predictive: [`multivariate_normal_known_covariance_predictive`][multivariate_normal_known_covariance_predictive] | - |
| Multivariate Normal (known precision) | [MultivariateNormal] | [`multivariate_normal_known_precision`][multivariate_normal_known_precision] | Helper: [`multivariate_normal_known_precision_inputs`][multivariate_normal_known_precision_inputs]<br>Predictive: [`multivariate_normal_known_precision_predictive`][multivariate_normal_known_precision_predictive] | - |
| Multivariate Normal (known mean) | [InverseWishart] | [`multivariate_normal_known_mean`][multivariate_normal_known_mean] | Helper: [`multivariate_normal_known_mean_inputs`][multivariate_normal_known_mean_inputs]<br>Predictive: - | - |
| Multivariate Normal | [NormalInverseWishart] | [`multivariate_normal`][multivariate_normal] | Helper: [`multivariate_normal_inputs`][multivariate_normal_inputs]<br>Predictive: [`multivariate_normal_predictive`][multivariate_normal_predictive] | - |

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

<!-- Predictive Function Links -->
[binomial_beta_predictive]: models.md#conjugate.models.binomial_beta_predictive
[bernoulli_beta_predictive]: models.md#conjugate.models.bernoulli_beta_predictive
[negative_binomial_beta_predictive]: models.md#conjugate.models.negative_binomial_beta_predictive
[geometric_beta_predictive]: models.md#conjugate.models.geometric_beta_predictive
[categorical_dirichlet_predictive]: models.md#conjugate.models.categorical_dirichlet_predictive
[multinomial_dirichlet_predictive]: models.md#conjugate.models.multinomial_dirichlet_predictive
[poisson_gamma_predictive]: models.md#conjugate.models.poisson_gamma_predictive
[exponential_gamma_predictive]: models.md#conjugate.models.exponential_gamma_predictive
[gamma_known_shape_predictive]: models.md#conjugate.models.gamma_known_shape_predictive
[normal_known_variance_predictive]: models.md#conjugate.models.normal_known_variance_predictive
[normal_known_precision_predictive]: models.md#conjugate.models.normal_known_precision_predictive
[normal_known_mean_predictive]: models.md#conjugate.models.normal_known_mean_predictive
[normal_predictive]: models.md#conjugate.models.normal_predictive
[linear_regression_predictive]: models.md#conjugate.models.linear_regression_predictive
[multivariate_normal_known_covariance_predictive]: models.md#conjugate.models.multivariate_normal_known_covariance_predictive
[multivariate_normal_known_precision_predictive]: models.md#conjugate.models.multivariate_normal_known_precision_predictive
[multivariate_normal_predictive]: models.md#conjugate.models.multivariate_normal_predictive

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
