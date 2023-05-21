import inspect
import os
import tempfile
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # pragma: no cover

import numpy as np  # pragma: no cover
import pandas as pd  # pragma: no cover
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline  # pragma: no cover

from scikit_longitudinal.data_preparation import LongitudinalDataset  # pragma: no cover
from scikit_longitudinal.data_preparation.aggregation_function import AggrFunc
from scikit_longitudinal.data_preparation.merwav_time_minus import MerWavTimeMinus
from scikit_longitudinal.data_preparation.merwav_time_plus import MerWavTimePlus
from scikit_longitudinal.data_preparation.separate_waves import SepWav
from scikit_longitudinal.templates.custom_data_preparation_mixin import DataPreparationMixin


def feature_selection_default_callback(
        step_idx: int,
        dummy_longitudinal_dataset: LongitudinalDataset,
        y: Union[pd.Series, np.ndarray],
        name: str,
        transformer: TransformerMixin, ) -> Tuple[
    np.ndarray, List[List[Union[int, str]]], List[Union[int, str]], List[str]]:
    _y = y
    _step_idx = step_idx
    if name in {
        "CorrelationBasedFeatureSelectionPerGroup",
        "CorrelationBasedFeatureSelection",
    }:
        data = transformer.apply_selected_features_and_rename(
            dummy_longitudinal_dataset.data,
            None,
        )
        dummy_longitudinal_dataset.set_data(data)
        dummy_longitudinal_dataset.setup_features_group("elsa")
        return dummy_longitudinal_dataset.data.to_numpy(), \
            dummy_longitudinal_dataset.feature_groups(), \
            dummy_longitudinal_dataset.non_longitudinal_features(), \
            dummy_longitudinal_dataset.data.columns.tolist()
    return dummy_longitudinal_dataset.data.to_numpy(), \
        dummy_longitudinal_dataset.feature_groups(), \
        dummy_longitudinal_dataset.non_longitudinal_features(), \
        dummy_longitudinal_dataset.data.columns.tolist()


def handle_errors(f):
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print(f"An error occurred in function {f.__name__}: {str(e)}")
            raise

    return wrapper


def validate_input(f: Callable) -> Callable:
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        y = None
        list_args = list(args)

        X = list_args[1]

        if X is None:
            raise ValueError(f"No data was passed to {f.__name__}.")
        if not isinstance(X, (np.ndarray, pd.DataFrame, list)):
            raise ValueError("Input data must be a numpy array, pandas DataFrame, or 2D list.")

        if len(list_args) > 2:
            y = list_args[2]
            if y is not None and not isinstance(y, (pd.Series, np.ndarray, list)):
                raise ValueError("y must be a pandas Series, numpy array, 2D list, or not passing any target data")

        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        list_args[1] = X

        if len(list_args) > 2 and y is not None:
            if isinstance(y, list):
                y = np.array(y)
            elif isinstance(y, pd.Series):
                y = y.to_numpy()
            list_args[2] = y

        return f(*list_args, **kwargs)

    return wrapper


# pylint: disable=W0511
class LongitudinalPipeline(Pipeline):
    def __init__(
            self,
            steps: List[Tuple[str, Any]],
            features_group: List[List[Union[int, str]]],
            non_longitudinal_features: List[Union[int, str]] = None,
            update_feature_groups_callback: Union[Callable, str] = None,
            feature_list_names: List[str] = None,
    ) -> None:
        super().__init__(steps=steps)
        self._longitudinal_data: np.ndarray = np.array([])
        self.features_group: List[List[Union[int, str]]] = features_group
        self.non_longitudinal_features: List[Union[int, str]] = non_longitudinal_features
        self.update_feature_groups_callback: Union[Callable, str] = update_feature_groups_callback
        self.feature_list_names: List[str] = feature_list_names
        self.selected_feature_indices_: np.ndarray = np.array([])

        if update_feature_groups_callback is not None:
            if update_feature_groups_callback == "default":
                self.update_feature_groups_callback = feature_selection_default_callback
                return

            if not callable(update_feature_groups_callback) or isinstance(
                update_feature_groups_callback, str
            ):
                raise ValueError("update_data_callback must be a callable function or a default string value.")

            sig = inspect.signature(update_feature_groups_callback)
            parameters = sig.parameters.values()
            parameter_count = len(parameters)

            if parameter_count != 5:
                raise ValueError(
                    f"update_data_callback must accept 5 parameters, got {parameter_count}.\n"
                    "The parameters are: \n"
                    "* step_idx (integer), \n"
                    "* dummy_longitudinal_dataset (LongitudinalDataset), \n"
                    "* y (Union[pd.Series, np.ndarray]), \n"
                    "* name (str), \n"
                    "* transformer (TransformerMixin)"
                )

            if any(
                    param.annotation != expected_type
                    for param, expected_type in zip(
                        parameters,
                        [
                            int,
                            LongitudinalDataset,
                            Union[pd.Series, np.ndarray],
                            str,
                            TransformerMixin,
                        ],
                    )
            ):
                raise ValueError(
                    "update_data_callback must accept the following parameters: \n"
                    "* step_idx (integer), \n"
                    "* dummy_longitudinal_dataset (LongitudinalDataset), \n"
                    "* y (Union[pd.Series, np.ndarray]), \n"
                    "* name (str), \n"
                    "* transformer (TransformerMixin). \n"
                    f"Got: {', '.join([param.annotation.__name__ for param in parameters])}"
                )

    @handle_errors
    @validate_input
    def fit(
            self,
            X: np.ndarray,
            y: Optional[Union[pd.Series, np.ndarray]] = None,
            **fit_params: Dict[str, Any],
    ) -> "LongitudinalPipeline":
        self._longitudinal_data = X.copy()
        self.selected_feature_indices_ = np.arange(X.shape[1])

        if y is not None:
            y = y.copy()

        filtered_steps = [(name, transformer) for name, transformer in self.steps[:-1] if transformer is not None]

        for step_idx, (name, transformer) in enumerate(filtered_steps):
            transformer = self._add_extra_longitudinal_parameters(transformer, name)
            if isinstance(transformer, (MerWavTimeMinus, MerWavTimePlus)):
                continue
            elif isinstance(transformer, AggrFunc):
                transformer.prepare_data(self._longitudinal_data, y)
                (
                    transformed_dataset,
                    self.features_group,
                    self.non_longitudinal_features,
                    self.feature_list_names,
                ) = transformer._transform()
                self.selected_feature_indices_ = np.array(
                    [transformed_dataset.columns.get_loc(indice) for indice in transformed_dataset.columns]
                )
                self._longitudinal_data = transformed_dataset
                continue
            elif isinstance(transformer, SepWav):
                continue
            else:
                X_transformed = transformer.fit_transform(self._longitudinal_data, y, **fit_params)

            if not hasattr(transformer, "selected_features_"):
                raise ValueError(f"Transformer {name} does not have a selected_features_ attribute.")

            self.selected_feature_indices_ = self.selected_feature_indices_[transformer.selected_features_]
            self.feature_list_names = [self.feature_list_names[i] for i in transformer.selected_features_]

            if hasattr(transformer, "features_group") and not (
                    hasattr(transformer, "cfs_type_") and transformer.cfs_type_ == "cfs"
            ):
                self._longitudinal_data = self._longitudinal_data[:, transformer.selected_features_]
            else:
                self._longitudinal_data = X_transformed

            (
                self._longitudinal_data,
                self.features_group,
                self.non_longitudinal_features,
                self.feature_list_names,
            ) = self._update_data_callback(name, step_idx, transformer, y)

        if self._final_estimator is not None:
            if hasattr(self._final_estimator, "features_group"):
                self._final_estimator.features_group = self.features_group
            if hasattr(self._final_estimator, "non_longitudinal_features"):
                self._final_estimator.non_longitudinal_features = self.non_longitudinal_features
            if hasattr(self._final_estimator, "feature_list_names"):
                self._final_estimator.feature_list_names = self.feature_list_names
            if isinstance(self.steps[-2][1], SepWav):
                if hasattr(self.steps[-2][1], "classifier"):
                    self.steps[-2][1].classifier = self._final_estimator
                else:
                    raise ValueError("SepWav does not have a classifier attribute.")
                self.steps[-2][1].fit(self._longitudinal_data, y, **fit_params)
            else:
                self._final_estimator.fit(self._longitudinal_data, y, **fit_params)

        return self

    def _add_extra_longitudinal_parameters(self, transformer: TransformerMixin, name: str) -> TransformerMixin:
        if isinstance(transformer, DataPreparationMixin):
            transformer.features_group = self.features_group
            transformer.non_longitudinal_features = self.non_longitudinal_features
            transformer.feature_list_names = self.feature_list_names
        elif hasattr(transformer, "features_group"):
            if self.update_feature_groups_callback is None:
                raise ValueError(
                    f"The transformer {name} has a features_group attribute, but no update_data"
                    "callback was passed to the pipeline. Please pass an update_data_callback"
                    "function, you to update not solely the selected features but also the non"
                    "longitudinal features."
                )
            if not callable(self.update_feature_groups_callback):
                raise ValueError("update_data_callback must be a callable function")
            transformer.features_group = self.features_group
            if hasattr(transformer, "non_longitudinal_features"):
                transformer.non_longitudinal_features = self.non_longitudinal_features
            if not self.feature_list_names:
                raise ValueError(
                    f"The features_group attribute of the transformer named {name} designates it as a "
                    "longitudinal-based transformer. But no feature_list_names were sent to the pipeline "
                    "as of yet. Please hand the names of the features."
                )
        return transformer

    def _update_data_callback(
            self, name: str, step_idx: int, transformer: TransformerMixin, y: Optional[Union[pd.Series, np.ndarray]]
    ) -> Tuple[np.ndarray, List[List[Union[int, str]]], List[Union[int, str]], List[str]]:
        df = pd.DataFrame(self._longitudinal_data, columns=self.feature_list_names)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmpfile:
            df.to_csv(tmpfile.name, index=False)

        dummy_longitudinal_dataset = LongitudinalDataset(file_path=tmpfile.name)
        dummy_longitudinal_dataset.load_data()
        (
            updated_longitudinal_data,
            updated_features_group,
            update_non_longitudinal_features,
            updated_feature_list_names,
        ) = self.update_feature_groups_callback(step_idx, dummy_longitudinal_dataset, y, name, transformer)

        os.remove(tmpfile.name)

        return (
            updated_longitudinal_data,
            updated_features_group,
            update_non_longitudinal_features,
            updated_feature_list_names,
        )

    @validate_input
    def predict(self, X: np.ndarray, **predict_params: Dict[str, Any]) -> np.ndarray:
        if X is None:
            raise ValueError("No data was passed to predict.")
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data must be a numpy array.")

        X = X[:, self.selected_feature_indices_]

        if isinstance(self.steps[-2][1], SepWav):
            return self.steps[-2][1].predict(X, **predict_params)
        else:
            return self._final_estimator.predict(X, **predict_params)

    @validate_input
    def transform(self, X: np.ndarray, **transform_params: Dict[str, Any]) -> np.ndarray:
        if X is None:
            raise ValueError("No data was passed to transform.")
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data must be a numpy array.")

        X = X[:, self.selected_feature_indices_]

        return self._final_estimator.transform(X, **transform_params)
