# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "conjugate-models",
#     "numpy==2.2.5",
# ]
# ///

import marimo

__generated_with = "0.13.7"
app = marimo.App(width="full", app_title="Available Rental Cars")


@app.cell
def _(mo):
    mo.md(
        r"""
    Find the original example on [Wikipedia](https://en.wikipedia.org/wiki/Conjugate_prior#Practical_example).

    Suppose a rental car service operates in your city. Drivers can drop off and pick up cars anywhere inside the city limits. You can find and rent cars using an app.

    Suppose you wish to find the probability that you can find a rental car within a short distance of your home address at any time of day.

    Over three days you look at the app and find the following number of cars within a short distance of your home address: $\mathbf{x} = [3, 4, 1]$

    We can represent this data with an numpy array or any other data that works like a [numerical array](./../generalized-inputs).

    ```python
    import numpy as np

    x = np.array([3, 4, 1])
    ```
    """
    )
    return


@app.cell
async def _():
    import micropip

    await micropip.install("conjugate-models")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np

    x = np.array([3, 4, 1])
    return mo, np, x


@app.cell
def _(mo, np, x):
    numerator = "+".join(map(str, x))
    den = len(x)
    lam = np.mean(x)
    at_least_one = 1 - np.exp(-lam)

    mo.md(rf"""
    Suppose we assume the data comes from a Poisson distribution. In that case, we can compute the maximum likelihood estimate of the parameters of the mode, which is

    $$
    \lambda = \frac{{ {numerator} }}{{ {den} }} \approx  {lam:.2f}
    $$

    Using this maximum likelihood estimate, we can compute the probability that there will be at least one car available on a given day:

    $$
    p(x > 0 | \lambda \approx {lam:.2f}) = 1 - p(x=0 | \lambda \approx {lam:.2f}) = 1 - \frac{{ {lam:.2f}^0 e^{{ -{lam:.2f} }} }}{{0!}} \approx {at_least_one:.2f}
    $$
    """)
    return


@app.cell
def _(mo):
    build_up = mo.md(r"""
    This is the Poisson distribution that is *the* most likely to have generated the observed data $\mathbf{x}$. But the data could also have come from another Poisson distribution, e.g., one with $\lambda = 3$, or $\lambda = 2$, etc. In fact, there is an infinite number of Poisson distributions that *could* have generated the observed data. With relatively few data points, we should be quite uncertain about which exact Poisson distribution generated this data. Intuitively we should instead take a weighted average of the the probability of $p(x > 0 | \lambda )$ for each of those Poisson distributions, weighted by how likely they each are, given the data we've observed $\mathbf{x}$.

    Generally, this quantity is known as the prior predictive distribution $p(x| \mathbf{x}) = \int_{\theta}{p(x|\theta)p(\theta| \mathbf{x})d\theta}$, where $x$ is a new data point, $\mathbf{x}$ is the observed data and $\theta$ are the parameters of the model. Using Bayes' theorem, we can expand

    Returning to our example, if we pick the Gamma distribution as our prior distribution over the rate of the Poisson distributions, then the posterior predictive is the negative binomial distribution. The Gamma distribution is parameterized by two hyperparameters $\alpha$, $\beta$, which we have to choose. By looking at plots of the Gamma distribution, we pick a reasonable prior for the average number of cars. The choice of prior hyperparameters is inherently subjective and based on prior knowledge.
    """)

    parameters = mo.md(r"""
    ### Prior Parameters
    $\alpha$ = {alpha}

    $\beta$ = {beta}
    """).batch(
        alpha=mo.ui.slider(start=0.01, stop=5, step=0.01, show_value=True, value=2),
        beta=mo.ui.slider(start=0.01, stop=5, step=0.01, show_value=True, value=2),
    )
    return build_up, parameters


@app.cell
def _(parameters, x):
    from conjugate.distributions import Gamma
    from conjugate.models import poisson_gamma, poisson_gamma_predictive

    prior = Gamma(**parameters.value)
    posterior = poisson_gamma(n=len(x), x_total=sum(x), prior=prior)
    posterior_predictive = poisson_gamma_predictive(distribution=posterior)
    return posterior, posterior_predictive, prior


@app.cell
def _(build_up, mo, parameters):
    mo.vstack([build_up, parameters])
    return


@app.cell
def _(prior, parameters):
    upper = 10
    ax = prior.set_bounds(0, upper).plot_pdf(label="prior")
    ax.set(
        title=f"Prior distribution for average\nGamma(alpha={parameters.value['alpha']}, beta={parameters.value['beta']})",
        xlabel="Average number of cars",
    )
    return


@app.cell
def _(mo, parameters):
    posterior_section = mo.md(r"""
    Given the prior hyperparameters $\alpha$ and $\beta$ we can compute the posterior hyperparameters $\alpha \prime = \alpha + \sum_{i} {x_i}$ and $\beta \prime = \beta + n$.
    """)

    conjugate_posterior_section = mo.md(f"""
    Using `conjugate-models`, we can import `poisson_gamma` to compute the posterior hyperparmeters instead of remembering the formula:

    ```python
    from conjugate.distributions import Gamma
    from conjugate.models import poisson_gamma

    n = len(x)
    x_total = sum(x)

    prior = Gamma(alpha={parameters.value["alpha"]}, beta={parameters.value["beta"]})
    posterior: Gamma = poisson_gamma(n=n, x_total=x_total, prior=prior)
    ```
    """)

    mo.vstack([posterior_section, conjugate_posterior_section])
    return


@app.cell
def _(mo, posterior, posterior_predictive):
    greater_than_zero = 1 - posterior_predictive.dist.cdf(0)

    posterior_predictive_section = mo.md(rf"""
    Given the posterior hyperparameters, we can finally compute the posterior predictive of

    $$
    p(x > 0| \mathbf{{x}}) = 1 - p(x=0 | \mathbf{{x}}) = 1 - NB(0 | {posterior_predictive.n.item():.2f}, \frac{{ {posterior.beta:.2f} }}{{ 1 + {posterior.beta:.2f} }}) \approx {greater_than_zero:.2f}
    $$

    With `conjugate-models`, the `poisson_gamma` function has complimentary `poisson_gamma_predictive` function for posterior
    predictive. The `dist` attribute to [access scipy distributions](./../scipy-connection). This gives
    access to various statistics and methods of the distribuition making this calculation
    trivial.

    ```python
    from conjugate.models import poisson_gamma_predictive

    posterior_predictive: "NegativeBinomial" = poisson_gamma_predictive(distribution=posterior)
    greater_than_zero = 1 - posterior_predictive.dist.cdf(0)
    # {greater_than_zero:.2f}
    ```

    This much more conservative estimate reflects the uncertainty in the model parameters, which the posterior predictive takes into account.
    """)

    mo.vstack(
        [
            posterior_predictive_section,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
