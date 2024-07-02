import inspect
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.preprocessors.feature_selection.correlation_feature_selection import (
    CorrelationBasedFeatureSelectionPerGroup,
)


def default_callback(
    step_idx: int,  # pylint: disable=W0613
    dummy_longitudinal_dataset: LongitudinalDataset,
    y: Union[pd.Series, np.ndarray],  # pylint: disable=W0613
    name: str,  # pylint: disable=W0613
    transformer: TransformerMixin,
) -> Tuple[np.ndarray, List[List[int]], List[Union[int, str]], List[str]]:
    """Default callback function for updating feature groups.

    Args:
        step_idx:
            The index of the current processing step.
        dummy_longitudinal_dataset:
            Dummy dataset object representing the current state of the data.
        y:
            Target data.
        name:
            Name of the transformer.`
        transformer:
            The transformer being processed.

    Returns:
        Tuple containing:
        - Numpy array representation of the updated data.
        - List representing the updated feature groups.
        - List of non-longitudinal features.
        - List of column names in the updated data.

    """
    if isinstance(transformer, CorrelationBasedFeatureSelectionPerGroup):
        data = transformer.apply_selected_features_and_rename(dummy_longitudinal_dataset.data, None)
        dummy_longitudinal_dataset.set_data(data)
        dummy_longitudinal_dataset.setup_features_group("elsa")

    return (
        dummy_longitudinal_dataset.data.to_numpy(),
        dummy_longitudinal_dataset.feature_groups(),
        dummy_longitudinal_dataset.non_longitudinal_features(),
        dummy_longitudinal_dataset.data.columns.tolist(),
    )


def validate_update_feature_groups_callback(callback: Callable) -> Callable:
    """Validate the update_feature_groups_callback function.

    Args:
        callback:
            The callback function to be validated. The callback should respect the following structure:
                def callback(step_idx: int,
                            dummy_longitudinal_dataset: LongitudinalDataset,
                            y: Union[pd.Series, np.ndarray],
                            name: str,
                            transformer: TransformerMixin) -> Tuple[np.ndarray, List[List[int]],
                                                                      List[Union[int, str]], List[str]]:
                ...
                Look at the default_callback function for an example.

    Returns:
        The validated callback function.

    Raises:
        ValueError: If the callback is not valid.

    """
    if callback == "default":
        return default_callback

    if not callable(callback) or isinstance(callback, str):
        raise ValueError("update_data_callback must be a callable function or a 'default' string value.")

    sig = inspect.signature(callback)
    parameters = list(sig.parameters.values())
    parameter_count = len(parameters)

    expected_params = [
        (int, "step_idx"),
        (LongitudinalDataset, "dummy_longitudinal_dataset"),
        (Union[pd.Series, np.ndarray], "y"),
        (str, "name"),
        (TransformerMixin, "transformer"),
    ]

    if parameter_count != len(expected_params):
        raise ValueError(f"update_data_callback must accept {len(expected_params)} parameters, got {parameter_count}.")

    for param, (expected_type, expected_name) in zip(parameters, expected_params):
        if param.annotation != expected_type or param.name != expected_name:
            raise ValueError(
                f"Expected parameter of type {expected_type.__name__} named {expected_name}, "
                f"got parameter of type {param.annotation.__name__} named {param.name}."
            )

    return callback
