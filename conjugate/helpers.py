"""Helper functions to extract sufficient statistics from observations.

These functions bridge the gap between raw data and the summary statistics
required by conjugate model functions.

Example:
    ```python
    from conjugate.distributions import Gamma
    from conjugate.models import poisson_gamma
    from conjugate.helpers import poisson_gamma_inputs

    observations = [3, 5, 2, 4, 1]
    prior = Gamma(1, 1)
    posterior = poisson_gamma(**poisson_gamma_inputs(observations), prior=prior)
    ```

For PyTensor compatibility:
    ```python
    from conjugate.helpers import poisson_gamma_inputs, pytensor_ops

    inputs = poisson_gamma_inputs(observations, **pytensor_ops)
    ```
"""

import numpy as np
from typing import Callable

# =============================================================================
# Default Operations (NumPy-compatible, handle scalars)
# =============================================================================


def _default_sum(x):
    """Sum elements. Handles both arrays and scalars."""
    return np.sum(x)


def _default_len(x):
    """Count elements. Handles both arrays and scalars via atleast_1d."""
    return np.atleast_1d(x).size


def _default_prod(x):
    """Product of elements. Handles both arrays and scalars."""
    return np.prod(x)


def _default_log(x):
    """Natural logarithm. Handles both arrays and scalars."""
    return np.log(x)


def _default_max(x):
    """Maximum value. Handles both arrays and scalars."""
    return np.max(x)


def _default_min(x):
    """Minimum value. Handles both arrays and scalars."""
    return np.min(x)


def _default_mean(x):
    """Mean of elements. Handles both arrays and scalars."""
    return np.mean(x)


def _default_cos(x):
    """Cosine. Handles both arrays and scalars."""
    return np.cos(x)


def _default_sin(x):
    """Sine. Handles both arrays and scalars."""
    return np.sin(x)


def _default_arctan2(y, x):
    """Two-argument arctangent. Handles both arrays and scalars."""
    return np.arctan2(y, x)


def _default_maximum(x, y):
    """Element-wise maximum. Handles both arrays and scalars."""
    return np.maximum(x, y)


def _default_reciprocal(x):
    """Element-wise reciprocal (1/x). Handles both arrays and scalars."""
    return 1.0 / x


def _default_power(x, exp):
    """Element-wise power. Handles both arrays and scalars."""
    return np.power(x, exp)


# =============================================================================
# Operation Bundles
# =============================================================================

numpy_ops = {
    "sum_fn": _default_sum,
    "len_fn": _default_len,
    "prod_fn": _default_prod,
    "log_fn": _default_log,
    "max_fn": _default_max,
    "min_fn": _default_min,
    "mean_fn": _default_mean,
    "cos_fn": _default_cos,
    "sin_fn": _default_sin,
    "arctan2_fn": _default_arctan2,
    "maximum_fn": _default_maximum,
    "reciprocal_fn": _default_reciprocal,
    "power_fn": _default_power,
    "asarray_fn": np.asarray,
}


class _PyTensorOpsProxy:
    """Lazy loader for PyTensor operations to avoid hard dependency."""

    _ops = None

    def _get_ops(self):
        """Lazy import PyTensor operations."""
        if self._ops is None:
            try:
                import pytensor.tensor as pt

                self._ops = {
                    "sum_fn": pt.sum,
                    "len_fn": lambda x: pt.atleast_1d(x).size,
                    "prod_fn": pt.prod,
                    "log_fn": pt.log,
                    "max_fn": pt.max,
                    "min_fn": pt.min,
                    "mean_fn": pt.mean,
                    "cos_fn": pt.cos,
                    "sin_fn": pt.sin,
                    "arctan2_fn": pt.arctan2,
                    "maximum_fn": pt.maximum,
                    "reciprocal_fn": pt.reciprocal,
                    "power_fn": pt.power,
                    "asarray_fn": pt.as_tensor_variable,
                }
            except ImportError:
                raise ImportError(
                    "PyTensor is not installed. Install with 'pip install pytensor' "
                    "to use pytensor_ops."
                )
        return self._ops

    def __getitem__(self, key):
        return self._get_ops()[key]

    def keys(self):
        return self._get_ops().keys()

    def values(self):
        return self._get_ops().values()

    def items(self):
        return self._get_ops().items()


pytensor_ops = _PyTensorOpsProxy()


# =============================================================================
# Category 1: Simple Sum/Count (x_total, n)
# Models: poisson_gamma, exponential_gamma, gamma_known_shape
# =============================================================================


def poisson_gamma_inputs(
    x,
    *,
    sum_fn: Callable | None = None,
    len_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for poisson_gamma model.

    Args:
        x: Observations (array-like or scalar)
        sum_fn: Function to sum elements, defaults to np.sum
        len_fn: Function to count elements, defaults to size via np.atleast_1d

    Returns:
        Dict with keys 'x_total' and 'n' for use with poisson_gamma()

    Example:
        ```python
        from conjugate.models import poisson_gamma
        from conjugate.helpers import poisson_gamma_inputs

        data = [3, 5, 2, 4]
        inputs = poisson_gamma_inputs(data)
        # inputs = {'x_total': 14, 'n': 4}
        posterior = poisson_gamma(**inputs, prior=prior)
        ```
    """
    if sum_fn is None:
        sum_fn = _default_sum
    if len_fn is None:
        len_fn = _default_len

    return {
        "x_total": sum_fn(x),
        "n": len_fn(x),
    }


# Alias - same sufficient statistics
exponential_gamma_inputs = poisson_gamma_inputs


def gamma_known_shape_inputs(
    x,
    *,
    sum_fn: Callable | None = None,
    len_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for gamma_known_shape model.

    Args:
        x: Observations (array-like or scalar)
        sum_fn: Function to sum elements, defaults to np.sum
        len_fn: Function to count elements, defaults to size via np.atleast_1d

    Returns:
        Dict with keys 'x_total' and 'n' for use with gamma_known_shape()
    """
    if sum_fn is None:
        sum_fn = _default_sum
    if len_fn is None:
        len_fn = _default_len

    return {
        "x_total": sum_fn(x),
        "n": len_fn(x),
    }


# =============================================================================
# Category 2: Success/Trial Counts (x, n)
# Models: binomial_beta, bernoulli_beta, negative_binomial_beta
# =============================================================================


def binomial_beta_inputs(
    x,
    n,
    *,
    sum_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for binomial_beta model.

    Args:
        x: Number of successes (array-like or scalar)
        n: Number of trials (array-like or scalar)
        sum_fn: Function to sum elements, defaults to np.sum

    Returns:
        Dict with keys 'x' and 'n' for use with binomial_beta()
    """
    if sum_fn is None:
        sum_fn = _default_sum

    return {
        "x": sum_fn(x),
        "n": sum_fn(n),
    }


def bernoulli_beta_inputs(
    x,
    *,
    sum_fn: Callable | None = None,
    len_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for bernoulli_beta model.

    For multiple Bernoulli trials, returns statistics compatible with binomial_beta model.
    For a single trial, use directly with bernoulli_beta (n will be 1).

    Args:
        x: Binary outcomes (0 or 1, array-like or scalar)
        sum_fn: Function to sum elements, defaults to np.sum
        len_fn: Function to count elements, defaults to size via np.atleast_1d

    Returns:
        Dict with keys 'x' and 'n' for use with binomial_beta() or bernoulli_beta()

    Example:
        ```python
        # Multiple Bernoulli trials - use with binomial_beta
        from conjugate.models import binomial_beta
        from conjugate.helpers import bernoulli_beta_inputs

        data = [1, 0, 1, 1, 0]  # 5 trials, 3 successes
        inputs = bernoulli_beta_inputs(data)
        # inputs = {'x': 3, 'n': 5}
        posterior = binomial_beta(**inputs, prior=prior)
        ```
    """
    if sum_fn is None:
        sum_fn = _default_sum
    if len_fn is None:
        len_fn = _default_len

    return {
        "x": sum_fn(x),
        "n": len_fn(x),
    }


def negative_binomial_beta_inputs(
    x,
    r,
    *,
    sum_fn: Callable | None = None,
    len_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for negative_binomial_beta model.

    Args:
        x: Number of successes per experiment (array-like or scalar)
        r: Known number of failures per experiment (constant across all experiments)
        sum_fn: Function to sum elements, defaults to np.sum
        len_fn: Function to count elements, defaults to size via np.atleast_1d

    Returns:
        Dict with keys 'x', 'n', and 'r' for use with negative_binomial_beta()

    Example:
        ```python
        from conjugate.models import negative_binomial_beta
        from conjugate.helpers import negative_binomial_beta_inputs

        successes = [2, 3, 1]  # successes in 3 experiments
        failures_per_exp = 5  # known failures per experiment
        inputs = negative_binomial_beta_inputs(successes, failures_per_exp)
        # inputs = {'x': 6, 'n': 3, 'r': 5}
        posterior = negative_binomial_beta(**inputs, prior=prior)
        ```
    """
    if sum_fn is None:
        sum_fn = _default_sum
    if len_fn is None:
        len_fn = _default_len

    return {
        "x": sum_fn(x),
        "n": len_fn(x),
        "r": r,
    }


def geometric_beta_inputs(
    x,
    *,
    sum_fn: Callable | None = None,
    len_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for geometric_beta model.

    Args:
        x: Observations (array-like or scalar)
        sum_fn: Function to sum elements, defaults to np.sum
        len_fn: Function to count elements, defaults to size via np.atleast_1d

    Returns:
        Dict with keys 'x_total' and 'n' for use with geometric_beta()
    """
    if sum_fn is None:
        sum_fn = _default_sum
    if len_fn is None:
        len_fn = _default_len

    return {
        "x_total": sum_fn(x),
        "n": len_fn(x),
    }


def hypergeometric_beta_binomial_inputs(
    x,
    *,
    sum_fn: Callable | None = None,
    len_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for hypergeometric_beta_binomial model.

    Args:
        x: Observations (array-like or scalar)
        sum_fn: Function to sum elements, defaults to np.sum
        len_fn: Function to count elements, defaults to size via np.atleast_1d

    Returns:
        Dict with keys 'x_total' and 'n' for use with hypergeometric_beta_binomial()
    """
    if sum_fn is None:
        sum_fn = _default_sum
    if len_fn is None:
        len_fn = _default_len

    return {
        "x_total": sum_fn(x),
        "n": len_fn(x),
    }


# =============================================================================
# Category 3: Dirichlet/Multinomial (x = counts)
# Models: categorical_dirichlet, multinomial_dirichlet
# =============================================================================


def categorical_dirichlet_inputs(x) -> dict:
    """Extract sufficient statistics for categorical_dirichlet model.

    Args:
        x: Observed counts for each category (array-like)

    Returns:
        Dict with key 'x' for use with categorical_dirichlet()
    """
    return {"x": x}


def multinomial_dirichlet_inputs(x) -> dict:
    """Extract sufficient statistics for multinomial_dirichlet model.

    Args:
        x: Observed counts for each category (array-like)

    Returns:
        Dict with key 'x' for use with multinomial_dirichlet()
    """
    return {"x": x}


# =============================================================================
# Category 4: Sum and Sum of Squares (x_total, x2_total, n)
# Models: normal, normal_known_mean
# =============================================================================


def normal_inputs(
    x,
    *,
    sum_fn: Callable | None = None,
    len_fn: Callable | None = None,
    power_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for normal model.

    Works with: normal

    Args:
        x: Observations (array-like or scalar)
        sum_fn: Function to sum elements, defaults to np.sum
        len_fn: Function to count elements, defaults to size via np.atleast_1d
        power_fn: Function for element-wise power, defaults to np.power

    Returns:
        Dict with keys 'x_total', 'x2_total', and 'n'
    """
    if sum_fn is None:
        sum_fn = _default_sum
    if len_fn is None:
        len_fn = _default_len
    if power_fn is None:
        power_fn = _default_power

    return {
        "x_total": sum_fn(x),
        "x2_total": sum_fn(power_fn(x, 2)),
        "n": len_fn(x),
    }


def normal_known_mean_inputs(
    x,
    *,
    sum_fn: Callable | None = None,
    len_fn: Callable | None = None,
    power_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for normal_known_mean model.

    Args:
        x: Observations (array-like or scalar)
        sum_fn: Function to sum elements, defaults to np.sum
        len_fn: Function to count elements, defaults to size via np.atleast_1d
        power_fn: Function for element-wise power, defaults to np.power

    Returns:
        Dict with keys 'x_total', 'x2_total', and 'n'
    """
    if sum_fn is None:
        sum_fn = _default_sum
    if len_fn is None:
        len_fn = _default_len
    if power_fn is None:
        power_fn = _default_power

    return {
        "x_total": sum_fn(x),
        "x2_total": sum_fn(power_fn(x, 2)),
        "n": len_fn(x),
    }


def normal_known_variance_inputs(
    x,
    *,
    sum_fn: Callable | None = None,
    len_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for normal_known_variance model.

    Args:
        x: Observations (array-like or scalar)
        sum_fn: Function to sum elements, defaults to np.sum
        len_fn: Function to count elements, defaults to size via np.atleast_1d

    Returns:
        Dict with keys 'x_total' and 'n'
    """
    if sum_fn is None:
        sum_fn = _default_sum
    if len_fn is None:
        len_fn = _default_len

    return {
        "x_total": sum_fn(x),
        "n": len_fn(x),
    }


def normal_known_precision_inputs(
    x,
    *,
    sum_fn: Callable | None = None,
    len_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for normal_known_precision model.

    Args:
        x: Observations (array-like or scalar)
        sum_fn: Function to sum elements, defaults to np.sum
        len_fn: Function to count elements, defaults to size via np.atleast_1d

    Returns:
        Dict with keys 'x_total' and 'n'
    """
    if sum_fn is None:
        sum_fn = _default_sum
    if len_fn is None:
        len_fn = _default_len

    return {
        "x_total": sum_fn(x),
        "n": len_fn(x),
    }


# =============================================================================
# Category 5: Log Normal (ln_x_total, ln_x2_total, n)
# Models: log_normal
# =============================================================================


def log_normal_inputs(
    x,
    *,
    sum_fn: Callable | None = None,
    len_fn: Callable | None = None,
    log_fn: Callable | None = None,
    power_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for log_normal model.

    Args:
        x: Observations (array-like or scalar)
        sum_fn: Function to sum elements, defaults to np.sum
        len_fn: Function to count elements, defaults to size via np.atleast_1d
        log_fn: Function for natural log, defaults to np.log
        power_fn: Function for element-wise power, defaults to np.power

    Returns:
        Dict with keys 'ln_x_total', 'ln_x2_total', and 'n'
    """
    if sum_fn is None:
        sum_fn = _default_sum
    if len_fn is None:
        len_fn = _default_len
    if log_fn is None:
        log_fn = _default_log
    if power_fn is None:
        power_fn = _default_power

    ln_x = log_fn(x)

    return {
        "ln_x_total": sum_fn(ln_x),
        "ln_x2_total": sum_fn(power_fn(ln_x, 2)),
        "n": len_fn(x),
    }


# =============================================================================
# Category 6: Products (x_prod, n)
# Models: gamma_known_rate
# =============================================================================


def gamma_known_rate_inputs(
    x,
    *,
    prod_fn: Callable | None = None,
    len_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for gamma_known_rate model.

    Args:
        x: Observations (array-like or scalar)
        prod_fn: Function to take product of elements, defaults to np.prod
        len_fn: Function to count elements, defaults to size via np.atleast_1d

    Returns:
        Dict with keys 'x_prod' and 'n'
    """
    if prod_fn is None:
        prod_fn = _default_prod
    if len_fn is None:
        len_fn = _default_len

    return {
        "x_prod": prod_fn(x),
        "n": len_fn(x),
    }


# =============================================================================
# Category 7: Sum and Product (x_total, x_prod, n)
# Models: gamma
# =============================================================================


def gamma_inputs(
    x,
    *,
    sum_fn: Callable | None = None,
    prod_fn: Callable | None = None,
    len_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for gamma model.

    Args:
        x: Observations (array-like or scalar)
        sum_fn: Function to sum elements, defaults to np.sum
        prod_fn: Function to take product of elements, defaults to np.prod
        len_fn: Function to count elements, defaults to size via np.atleast_1d

    Returns:
        Dict with keys 'x_total', 'x_prod', and 'n'
    """
    if sum_fn is None:
        sum_fn = _default_sum
    if prod_fn is None:
        prod_fn = _default_prod
    if len_fn is None:
        len_fn = _default_len

    return {
        "x_total": sum_fn(x),
        "x_prod": prod_fn(x),
        "n": len_fn(x),
    }


# =============================================================================
# Category 8: Products of x and (1-x) (x_prod, one_minus_x_prod, n)
# Models: beta
# =============================================================================


def beta_inputs(
    x,
    *,
    prod_fn: Callable | None = None,
    len_fn: Callable | None = None,
    asarray_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for beta model.

    Args:
        x: Observations (array-like or scalar)
        prod_fn: Function to take product of elements, defaults to np.prod
        len_fn: Function to count elements, defaults to size via np.atleast_1d
        asarray_fn: Function to convert to array, defaults to np.asarray

    Returns:
        Dict with keys 'x_prod', 'one_minus_x_prod', and 'n'
    """
    if prod_fn is None:
        prod_fn = _default_prod
    if len_fn is None:
        len_fn = _default_len
    if asarray_fn is None:
        asarray_fn = np.asarray

    # Convert to array first to handle list inputs properly
    x_arr = asarray_fn(x)

    return {
        "x_prod": prod_fn(x_arr),
        "one_minus_x_prod": prod_fn(1 - x_arr),
        "n": len_fn(x),
    }


# =============================================================================
# Category 9: Log Sums with Optional Known Parameter (ln_x_total, n)
# Models: pareto_gamma
# =============================================================================


def pareto_gamma_inputs(
    x,
    *,
    x_m: float | None = None,
    sum_fn: Callable | None = None,
    len_fn: Callable | None = None,
    log_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for pareto_gamma model.

    Args:
        x: Observations (array-like or scalar)
        x_m: Known minimum value. If provided, returns ready-to-use dict.
             If None, returns raw ln_x_total (user must handle x_m separately).
        sum_fn: Function to sum elements, defaults to np.sum
        len_fn: Function to count elements, defaults to size via np.atleast_1d
        log_fn: Function for natural log, defaults to np.log

    Returns:
        Dict with keys 'ln_x_total', 'n', and 'x_m' (if provided)
    """
    if sum_fn is None:
        sum_fn = _default_sum
    if len_fn is None:
        len_fn = _default_len
    if log_fn is None:
        log_fn = _default_log

    n = len_fn(x)
    ln_x_total = sum_fn(log_fn(x))

    result = {"n": n, "ln_x_total": ln_x_total}

    if x_m is not None:
        result["x_m"] = x_m

    return result


# =============================================================================
# Category 10: Max/Min (x_max, n)
# Models: uniform_pareto
# =============================================================================


def uniform_pareto_inputs(
    x,
    *,
    max_fn: Callable | None = None,
    len_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for uniform_pareto model.

    Args:
        x: Observations (array-like or scalar)
        max_fn: Function to find maximum, defaults to np.max
        len_fn: Function to count elements, defaults to size via np.atleast_1d

    Returns:
        Dict with keys 'x_max' and 'n'
    """
    if max_fn is None:
        max_fn = _default_max
    if len_fn is None:
        len_fn = _default_len

    return {
        "x_max": max_fn(x),
        "n": len_fn(x),
    }


# =============================================================================
# Category 11: Reciprocal Sum (reciprocal_x_total, n)
# Models: inverse_gamma_known_rate
# =============================================================================


def inverse_gamma_known_rate_inputs(
    x,
    *,
    sum_fn: Callable | None = None,
    len_fn: Callable | None = None,
    reciprocal_fn: Callable | None = None,
    asarray_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for inverse_gamma_known_rate model.

    Args:
        x: Observations (array-like or scalar)
        sum_fn: Function to sum elements, defaults to np.sum
        len_fn: Function to count elements, defaults to size via np.atleast_1d
        reciprocal_fn: Function for reciprocal (1/x), defaults to lambda x: 1.0/x
        asarray_fn: Function to convert to array, defaults to np.asarray

    Returns:
        Dict with keys 'reciprocal_x_total' and 'n'
    """
    if sum_fn is None:
        sum_fn = _default_sum
    if len_fn is None:
        len_fn = _default_len
    if reciprocal_fn is None:
        reciprocal_fn = _default_reciprocal
    if asarray_fn is None:
        asarray_fn = np.asarray

    # Convert to array first to handle list inputs properly
    x_arr = asarray_fn(x)

    return {
        "reciprocal_x_total": sum_fn(reciprocal_fn(x_arr)),
        "n": len_fn(x),
    }


# =============================================================================
# Category 12: Von Mises - Known Concentration (cos_total, sin_total, n)
# Models: von_mises_known_concentration
# =============================================================================


def von_mises_known_concentration_inputs(
    x,
    *,
    sum_fn: Callable | None = None,
    len_fn: Callable | None = None,
    cos_fn: Callable | None = None,
    sin_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for von_mises_known_concentration model.

    Args:
        x: Observations - angular data (array-like or scalar)
        sum_fn: Function to sum elements, defaults to np.sum
        len_fn: Function to count elements, defaults to size via np.atleast_1d
        cos_fn: Function for cosine, defaults to np.cos
        sin_fn: Function for sine, defaults to np.sin

    Returns:
        Dict with keys 'cos_total', 'sin_total', and 'n'
    """
    if sum_fn is None:
        sum_fn = _default_sum
    if len_fn is None:
        len_fn = _default_len
    if cos_fn is None:
        cos_fn = _default_cos
    if sin_fn is None:
        sin_fn = _default_sin

    return {
        "cos_total": sum_fn(cos_fn(x)),
        "sin_total": sum_fn(sin_fn(x)),
        "n": len_fn(x),
    }


# =============================================================================
# Category 13: Von Mises - Known Direction (centered_cos_total, n)
# Models: von_mises_known_direction
# =============================================================================


def von_mises_known_direction_inputs(
    x,
    mu,
    *,
    sum_fn: Callable | None = None,
    len_fn: Callable | None = None,
    cos_fn: Callable | None = None,
    asarray_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for von_mises_known_direction model.

    Args:
        x: Observations - angular data (array-like or scalar)
        mu: Known direction parameter
        sum_fn: Function to sum elements, defaults to np.sum
        len_fn: Function to count elements, defaults to size via np.atleast_1d
        cos_fn: Function for cosine, defaults to np.cos
        asarray_fn: Function to convert to array, defaults to np.asarray

    Returns:
        Dict with keys 'centered_cos_total' and 'n'
    """
    if sum_fn is None:
        sum_fn = _default_sum
    if len_fn is None:
        len_fn = _default_len
    if cos_fn is None:
        cos_fn = _default_cos
    if asarray_fn is None:
        asarray_fn = np.asarray

    x = asarray_fn(x)

    return {
        "centered_cos_total": sum_fn(cos_fn(x - mu)),
        "n": len_fn(x),
    }


# =============================================================================
# Category 14: Weibull with Known Shape (x_beta_total, n)
# Models: weibull_inverse_gamma_known_shape
# =============================================================================


def weibull_inverse_gamma_known_shape_inputs(
    x,
    beta,
    *,
    sum_fn: Callable | None = None,
    len_fn: Callable | None = None,
    power_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for weibull_inverse_gamma_known_shape model.

    Args:
        x: Observations (array-like or scalar)
        beta: Known shape parameter
        sum_fn: Function to sum elements, defaults to np.sum
        len_fn: Function to count elements, defaults to size via np.atleast_1d
        power_fn: Function for element-wise power, defaults to np.power

    Returns:
        Dict with keys 'x_beta_total' and 'n'
    """
    if sum_fn is None:
        sum_fn = _default_sum
    if len_fn is None:
        len_fn = _default_len
    if power_fn is None:
        power_fn = _default_power

    return {
        "x_beta_total": sum_fn(power_fn(x, beta)),
        "n": len_fn(x),
    }


# =============================================================================
# Category 15: Multivariate (x_bar, n or just arrays)
# Models: multivariate_normal_known_covariance, multivariate_normal_known_precision
# =============================================================================


def multivariate_normal_known_covariance_inputs(
    X,
    *,
    len_fn: Callable | None = None,
    mean_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for multivariate_normal_known_covariance model.

    Args:
        X: Array of multivariate observations
        len_fn: Function to compute length, defaults to x.shape[0]
        mean_fn: Function to compute mean along axis 0, defaults to np.mean

    Returns:
        Dict with keys 'x_bar' and 'n'
    """
    if len_fn is None:

        def default_len_fn(x):
            return x.shape[0]  # Number of rows

        len_fn = default_len_fn
    if mean_fn is None:

        def default_mean_fn(x):
            return np.mean(x, axis=0)

        mean_fn = default_mean_fn

    return {
        "x_bar": mean_fn(X),
        "n": len_fn(X),
    }


def multivariate_normal_known_precision_inputs(
    X,
    *,
    len_fn: Callable | None = None,
    mean_fn: Callable | None = None,
) -> dict:
    """Extract sufficient statistics for multivariate_normal_known_precision model.

    Args:
        X: Observations matrix (n_samples x n_features)
        len_fn: Function to count samples (rows), defaults to len-like
        mean_fn: Function to compute mean along axis 0, defaults to np.mean

    Returns:
        Dict with keys 'x_bar' and 'n'
    """
    if len_fn is None:

        def default_len_fn(x):
            return x.shape[0]  # Number of rows

        len_fn = default_len_fn
    if mean_fn is None:

        def default_mean_fn(x):
            return np.mean(x, axis=0)

        mean_fn = default_mean_fn

    return {
        "x_bar": mean_fn(X),
        "n": len_fn(X),
    }


# =============================================================================
# Category 16: Identity (raw arrays) - for API consistency
# Models: linear_regression, multivariate_normal, multivariate_normal_known_mean
# =============================================================================


def linear_regression_inputs(X, y) -> dict:
    """Identity helper for linear_regression model.

    Returns X and y unchanged for API consistency.

    Args:
        X: Design matrix
        y: Response vector

    Returns:
        Dict with keys 'X' and 'y'
    """
    return {"X": X, "y": y}


def multivariate_normal_inputs(X) -> dict:
    """Identity helper for multivariate_normal model.

    Args:
        X: Observations matrix

    Returns:
        Dict with key 'X'
    """
    return {"X": X}


def multivariate_normal_known_mean_inputs(X, mu) -> dict:
    """Identity helper for multivariate_normal_known_mean model.

    Args:
        X: Observations matrix
        mu: Known mean

    Returns:
        Dict with keys 'X' and 'mu'
    """
    return {"X": X, "mu": mu}
