import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
async def _():
    import marimo as mo

    import micropip

    await micropip.install("conjugate-models")

    from conjugate.interactive import (
        supported_distributions,
        lookup_model,
        lookup_model_by_predictive,
    )

    return (
        lookup_model,
        lookup_model_by_predictive,
        mo,
        supported_distributions,
    )


@app.function
def create_model_documentation_url(model_name: str) -> str:
    return f"https://williambdean.github.io/conjugate/models/#conjugate.models.{model_name}"


@app.cell
def _(mo, supported_distributions):
    names = list(supported_distributions.keys())
    distribution = mo.ui.dropdown(
        options=names,
        label="Distribution:",
        value=names[0],
    )
    is_cdf = mo.ui.checkbox(label="Plot CDF", value=False)
    mo.hstack([distribution, is_cdf], justify="start")
    return distribution, is_cdf


@app.cell
def _(mo):
    x_min = mo.ui.number(label="x min")
    x_max = mo.ui.number(label="x max")

    mo.hstack(
        [x_min, x_max],
        justify="start",
    )
    return x_max, x_min


@app.cell
def _(mo):
    def get_parameter_ui(value, name):
        if value is bool:
            return mo.ui.checkbox(value=True)

        kwargs = {"show_value": True}
        metadata = value.__metadata__
        if metadata == ("Real",):
            return mo.ui.slider(start=-10, value=0.0, stop=10, step=0.01, **kwargs)
        elif metadata == ("Positive", "Real"):
            return mo.ui.slider(start=0.01, value=1.0, step=0.01, stop=10, **kwargs)
        elif metadata == ("Natural",):
            return mo.ui.slider(start=1, value=10, stop=30, step=1, **kwargs)
        elif metadata == ("Probability",):
            return mo.ui.slider(start=0, value=0.5, stop=1, step=0.01, **kwargs)

    return (get_parameter_ui,)


@app.cell
def _(distribution, supported_distributions):
    dist = supported_distributions[distribution.value]
    parameters = dist.__annotations__
    parameters_meta = {name: value for name, value in parameters.items()}

    return dist, parameters_meta


@app.cell
def _(get_parameter_ui, mo, parameters_meta):
    parameters_ui = mo.ui.dictionary(
        {name: get_parameter_ui(value, name) for name, value in parameters_meta.items()}
    )
    return (parameters_ui,)


@app.cell
def _(parameters_ui):
    parameters_ui
    return


@app.cell
def _(dist, is_cdf, mo, parameters_ui, x_max, x_min):
    initialize_dist = dist(**parameters_ui.value)

    if is_cdf.value:
        method = "plot_cdf"
    elif hasattr(initialize_dist, "plot_pmf"):
        method = "plot_pmf"
    elif hasattr(initialize_dist, "plot_pdf"):
        method = "plot_pdf"

    def default(value, default_value):
        return value if value is not None else default_value

    def plot():
        """Plot the distribution."""
        title = f"{dist.__name__} Distribution"
        if is_cdf.value:
            title = f"{dist.__name__} Distribution CDF"

        try:
            return getattr(initialize_dist, method)().set(title=title)
        except Exception:
            initialize_dist.set_bounds(
                default(x_min.value, getattr(initialize_dist, "_min_value", -10)),
                default(x_max.value, getattr(initialize_dist, "_max_value", 10)),
            )

            try:
                return getattr(initialize_dist, method)().set(title=title)
            except Exception:
                return mo.md("The distribution couldn't be plotted.")

    return initialize_dist, plot


@app.cell
def _(
    distribution,
    initialize_dist,
    is_cdf,
    lookup_model,
    lookup_model_by_predictive,
    mo,
    parameters_ui,
    x_max,
):
    from conjugate.plot import DiscretePlotMixin

    formatted_parameters = ", ".join(
        f"{key}={value}" for key, value in parameters_ui.value.items()
    )

    if is_cdf.value:
        plot_method = "plot_cdf"
    elif isinstance(initialize_dist, DiscretePlotMixin):
        plot_method = "plot_pmf"
    else:
        plot_method = "plot_pdf"

    actual_min_value = initialize_dist.min_value
    try:
        actual_max_value = initialize_dist.max_value
    except ValueError:
        actual_max_value = getattr(initialize_dist, "_max_value", x_max.value)

    code = mo.md(f"""
    Recreate the plot with the following code:

    ```python
    from conjugate.distributions import {distribution.value}

    distribution = {distribution.value}({formatted_parameters})
    distribution.set_bounds({actual_min_value}, {actual_max_value})

    ax = distribution.{plot_method}()
    ax.set(title="{distribution.value} Distribution")
    ```
    """)

    def create_bullet_points(models, links):
        model_list = "\n".join(
            f"- [`{model}`]({link})\n" for model, link in zip(models, links)
        )
        return f"\n{model_list}"

    models = lookup_model(distribution.value)
    models_from_predictive = lookup_model_by_predictive(distribution.value)
    if isinstance(models, list):
        links = [create_model_documentation_url(model) for model in models]
        model_list = create_bullet_points(models, links)
        reference = mo.md(f"""
        Model this distribution using:

        {model_list}
        """)
    elif models:
        link = create_model_documentation_url(models)
        reference = mo.md(f"""

        Model this distribution using the [`{models}`]({link}) function.

        """)
    elif isinstance(models_from_predictive, list):
        links = [
            create_model_documentation_url(model) for model in models_from_predictive
        ]
        model_list = create_bullet_points(models_from_predictive, links)
        reference = mo.md(f"""
        Find this distribution as the predictive distribution of:

        {model_list}
        """)
    elif models_from_predictive:
        link = create_model_documentation_url(models_from_predictive)
        reference = mo.md(f"""

        Find this distribution as the predictive distribution of [`{models_from_predictive}`]({link}) function.

        """)
    else:
        reference = None
    return code, reference


@app.cell
def _(code, mo, plot, reference):
    mo.hstack(
        [
            plot(),
            code
            if reference is None
            else mo.vstack([code, reference], justify="start"),
        ],
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
