from types import UnionType
from typing import Any, Annotated, Iterable, get_origin

from conjugate import distributions, models
from conjugate.plot import PlotDistMixin


def all_annotations_are_annotated(annotations: Iterable[Any]) -> bool:
    return all(get_origin(value) is Annotated for value in annotations)


def has_built_in_plotting(distribution) -> bool:
    return issubclass(distribution, PlotDistMixin) and all_annotations_are_annotated(
        distribution.__annotations__.values()
    )


def get_classes(module):
    return [
        getattr(module, name)
        for name in dir(module)
        if isinstance(getattr(module, name), type)
        and getattr(module, name).__module__ == module.__name__
    ]


def get_functions(module):
    return [
        getattr(module, name)
        for name in dir(module)
        if callable(getattr(module, name))
        and getattr(module, name).__module__ == module.__name__
    ]


supported_distributions = {
    distribution.__name__: distribution
    for distribution in get_classes(distributions)
    if has_built_in_plotting(distribution)
}

model_functions = {
    model.__name__: model
    for model in get_functions(models)
    if hasattr(model, "associated_likelihood")
}
conjugate_models = {
    name: model
    for name, model in model_functions.items()
    if not name.endswith("_predictive")
}
predictive_functions = {
    name: model
    for name, model in model_functions.items()
    if name.endswith("_predictive")
}


def get_prior_distribution(func):
    return func.__annotations__.get("prior") or func.__annotations__.get("distribution")


def get_predictive_distribution(func):
    annotations = func.__annotations__
    if "distribution" not in annotations:
        raise ValueError("Function does not have a 'distribution' annotation.")

    return annotations["return"]


def get_likelihood_distribution(func):
    return func.associated_likelihood


def get_associated_functions(name: str):
    return {"as_prior": ..., "as_likelihood": ...}


priors = {name: get_prior_distribution(func) for name, func in conjugate_models.items()}

likelihoods = {
    name: get_likelihood_distribution(func) for name, func in conjugate_models.items()
}
predictive_distributions = {
    name.replace("_predictive", ""): get_predictive_distribution(func)
    for name, func in predictive_functions.items()
}


def lookup_model(name: str) -> str | list[str] | None:
    models = [
        function_name
        for function_name, class_name in likelihoods.items()
        if class_name.__name__ == name
    ]

    if not models:
        return None

    if len(models) > 1:
        return models

    return models[0]


def lookup_prior(name: str):
    model = lookup_model(name)

    if model is None:
        return None

    return priors[model]


def get_associated_conjugate_models(name: str):
    model_names = [
        model_name
        for model_name, prior in priors.items()
        if not isinstance(prior, UnionType) and prior.__name__ == name
    ]
    return {
        name: distribution
        for name, distribution in likelihoods.items()
        if name in model_names
    }


def lookup_model_by_predictive(name: str) -> str | list[str] | None:
    model = [
        function_name
        for function_name, class_name in predictive_distributions.items()
        if not isinstance(class_name, UnionType) and class_name.__name__ == name
    ]

    if not model:
        return None

    if len(model) > 1:
        return model

    return model[0]
