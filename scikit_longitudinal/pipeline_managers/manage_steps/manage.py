from typing import Any, Callable, Dict, List, Tuple, Union, Optional

import numpy as np

from scikit_longitudinal.pipeline_managers.manage_steps.special_handler import SPECIAL_HANDLERS
from scikit_longitudinal.pipeline_managers.utils import configure_transformer
from scikit_longitudinal.data_preparation.aggregation_function import AggrFunc


def configure_and_fit_transformer(
    transformer: Any,
    name: str,
    X: np.ndarray,
    y: Optional[np.ndarray],
    fit_params: Dict[str, Any],
    selected_feature_indices: np.ndarray,
    feature_list_names: List[str],
    features_group: List[List[int]],
    non_longitudinal_features: List[Union[int, str]],
    update_feature_groups_callback: Callable,
) -> Tuple[Any, np.ndarray, Optional[np.ndarray], np.ndarray, List[str]]:
    """Configure and fit a given transformer.

    Args:
        transformer:
            The transformer to be configured and fitted.
        name:
            Name of the transformer.
        X:
            A numpy array of shape (n_samples, n_features) representing the input data.
        y:
            A numpy array representing the target values. Optional.
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
        Tuple containing:
            - The configured transformer
            - Transformed input data
            - Transformed target values (if applicable)
            - Selected feature indices
            - Updated feature list names
    Raises:
        ValueError:
            - If the transformer requires target labels 'y' for resampling but 'y' is
            None.
            - If the transformer does not have a 'selected_features_' attribute after fitting.
            - If the transformer is a longitudinal-based transformer but no feature list names
            were provided.
            - If the transformer requires 'features_group' but it is not provided.
            - If the callback function for updating feature groups is not provided for transformers
            that require it.
    """
    transformer = configure_transformer(
        transformer=transformer,
        name=name,
        features_group=features_group,
        non_longitudinal_features=non_longitudinal_features,
        feature_list_names=feature_list_names,
        update_feature_groups_callback=update_feature_groups_callback,
    )
    def is_instance_of_class_name(obj, name):
        for cls in obj.__class__.__mro__:
            if cls.__name__ == name:
                return True
        return False
    is_special_preprocessor = (
        is_instance_of_class_name(transformer, 'MerWavTimeMinus') or
        is_instance_of_class_name(transformer, 'MerWavTimePlus') or
        is_instance_of_class_name(transformer, 'AggrFunc') or
        is_instance_of_class_name(transformer, 'SepWav')
    )
    is_resampler = hasattr(transformer, 'fit_resample') and callable(getattr(transformer, 'fit_resample'))
    if is_resampler:
        if y is None:
            raise ValueError(f"Resampler '{name}' requires target labels 'y' for resampling.")
        X, y = transformer.fit_resample(X, y)
    if is_instance_of_class_name(transformer, 'AggrFunc') and (handler := SPECIAL_HANDLERS.get(AggrFunc)):
        transformer, X, y, selected_feature_indices, feature_list_names = handler.handle_transform(transformer, X, y)
    elif not is_special_preprocessor and not is_resampler:
        X_transformed = transformer.fit_transform(X, y)
        if not hasattr(transformer, "selected_features_"):
            raise ValueError(f"Transformer {name} does not have a selected_features_ attribute.")
        selected_feature_indices = transformer.selected_features_
        feature_list_names = [feature_list_names[i] for i in transformer.selected_features_]

        if handler := SPECIAL_HANDLERS.get(type(transformer)):
            transformer, X, y = handler.handle_transform(transformer, X, y)
        else:
            X = X_transformed
    return transformer, X, y, selected_feature_indices, feature_list_names


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
        final_estimator.non_longitudinal_features = non_longitudinal_features or []
    if hasattr(final_estimator, "feature_list_names"):
        final_estimator.feature_list_names = feature_list_names

    if handler := SPECIAL_HANDLERS.get(type(steps[-2][1])):
        final_estimator, steps, X, y = handler.handle_final_estimator(final_estimator, steps, X, y)

    final_estimator.fit(X, y, **fit_params)
    return final_estimator
