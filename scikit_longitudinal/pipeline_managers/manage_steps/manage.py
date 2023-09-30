from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np

from scikit_longitudinal.data_preparation import AggrFunc, MerWavTimeMinus, MerWavTimePlus, SepWav
from scikit_longitudinal.pipeline_managers.manage_steps.special_handler import SPECIAL_HANDLERS
from scikit_longitudinal.pipeline_managers.utils import configure_transformer


def configure_and_fit_transformer(
    transformer: Any,
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    fit_params: Dict[str, Any],
    selected_feature_indices: np.ndarray,
    feature_list_names: List[str],
    features_group: List[List[int]],
    non_longitudinal_features: List[Union[int, str]],
    update_feature_groups_callback: Callable,
) -> Tuple[Any, np.ndarray, np.ndarray, List[str]]:
    """Configure and fit a given transformer.

    Args:
        transformer:
            The transformer to be configured and fitted.
        name:
            Name of the transformer.
        X:
            A numpy array of shape (n_samples, n_features) representing the input data.
        y:
            A numpy array representing the target values.
        fit_params:
            Dictionary containing parameters to be used during the fit process.
        selected_feature_indices:
            Numpy array indicating the indices of the selected features.
        feature_list_names:
            List of names corresponding to the features.
        features_group:
            Grouping of the Longitudinal features.
        non_longitudinal_features:
            List of non-longitudinal features.
        update_feature_groups_callback:
            Callback function to update feature groups.

    Returns:
        Tuple containing the configured and fitted transformer, the transformed data X,
        updated selected_feature_indices, and updated feature_list_names.

    """
    transformer = configure_transformer(
        transformer=transformer,
        name=name,
        features_group=features_group,
        non_longitudinal_features=non_longitudinal_features,
        feature_list_names=feature_list_names,
        update_feature_groups_callback=update_feature_groups_callback,
    )

    if not isinstance(transformer, (MerWavTimeMinus, MerWavTimePlus, AggrFunc, SepWav)):
        X_transformed = transformer.fit_transform(X, y, **fit_params)

        if not hasattr(transformer, "selected_features_"):
            raise ValueError(f"Transformer {name} does not have a selected_features_ attribute.")

        selected_feature_indices = selected_feature_indices[transformer.selected_features_]
        feature_list_names = [feature_list_names[i] for i in transformer.selected_features_]

        if handler := SPECIAL_HANDLERS.get(type(transformer)):
            transformer, X, y = handler.handle_transform(transformer, X, y)
        else:
            X = X_transformed

    return transformer, X, selected_feature_indices, feature_list_names


def handle_final_estimator(
    final_estimator: Any,
    steps: List[Tuple[str, Any]],
    features_group: List[List[int]],
    non_longitudinal_features: List[Union[int, str]],
    feature_list_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    fit_params: Dict[str, Any],
) -> Any:
    """Configure and fit the final estimator in the pipeline.

    Args:
        final_estimator:
            The final estimator to be fitted.
        steps:
            List of steps in the pipeline.
        features_group:
            Grouping of the Longitudinal features.
        non_longitudinal_features:
            List of non-longitudinal features.
        feature_list_names:
            List of names corresponding to the features.
        X:
            A numpy array of shape (n_samples, n_features) representing the input data.
        y:
            A numpy array representing the target values.
        fit_params:
            Dictionary containing parameters to be used during the fit process.

    Returns:
        The fitted final_estimator.

    """
    if hasattr(final_estimator, "features_group"):
        final_estimator.features_group = features_group
    if hasattr(final_estimator, "non_longitudinal_features"):
        final_estimator.non_longitudinal_features = non_longitudinal_features
    if hasattr(final_estimator, "feature_list_names"):
        final_estimator.feature_list_names = feature_list_names

    if handler := SPECIAL_HANDLERS.get(type(steps[-2][1])):
        final_estimator, steps, X, y = handler.handle_final_estimator(final_estimator, steps, X, y)

    final_estimator.fit(X, y, **fit_params)
    return final_estimator
