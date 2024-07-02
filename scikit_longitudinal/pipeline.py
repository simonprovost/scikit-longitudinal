from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # pragma: no cover

import numpy as np  # pragma: no cover
import pandas as pd  # pragma: no cover
from rich import print  # pylint: disable=W0622
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline  # pragma: no cover

from scikit_longitudinal.data_preparation import LongitudinalDataset  # pragma: no cover
from scikit_longitudinal.pipeline_managers.manage_callbacks.manage import validate_update_feature_groups_callback
from scikit_longitudinal.pipeline_managers.manage_errors.manage import handle_errors, validate_input
from scikit_longitudinal.pipeline_managers.manage_steps import configure_and_fit_transformer, handle_final_estimator


# pylint: disable=W0511
class LongitudinalPipeline(Pipeline):
    """Custom pipeline for handling and processing longitudinal techniques (preprocessors, classifier, etc.).

    ⚠️ Scikit-Longitudinal's docstrings will be updated to reflect the most recent documentation available on Github.
    If something is inconsistent, consult the documentation first, then file an issue. ⚠️

    This class extends the scikit-learn's Pipeline to offer specialised methods and
    attributes for working with longitudinal data. It ensures that the longitudinal features
    and their structure are updated throughout the pipeline's transformations.

    Attributes:
        _longitudinal_data (np.ndarray):
            Longitudinal data being processed.
        features_group (List[List[int]]):
            Grouping of the Longitudinal features.
        non_longitudinal_features (List[Union[int, str]]):
            List of non-longitudinal features.
        feature_list_names (List[str]):
            List of names corresponding to the features.
        selected_feature_indices_ (np.ndarray):
            Indices of the selected features.
        final_estimator (Any):
            Final step in the pipeline.
        update_feature_groups_callback (Callable, optional):
            Callback function to update feature groups.

    Note:
        While this class maintains the interface of scikit-learn's Pipeline, it includes
        specific methods and validations to ensure the correct processing of longitudinal data.

    """

    def __init__(
        self,
        steps: List[Tuple[str, Any]],
        features_group: List[List[int]],
        non_longitudinal_features: List[Union[int, str]] = None,
        update_feature_groups_callback: Union[Callable, str] = None,
        feature_list_names: List[str] = None,
    ) -> None:
        super().__init__(steps=steps)
        self._longitudinal_data: np.ndarray = np.array([])
        self.features_group: List[List[int]] = features_group
        self.non_longitudinal_features: List[Union[int, str]] = non_longitudinal_features
        self.feature_list_names: List[str] = feature_list_names
        self.selected_feature_indices_: np.ndarray = np.array([])
        self.final_estimator = self.steps[-1][1]

        if update_feature_groups_callback is not None:
            self.update_feature_groups_callback = validate_update_feature_groups_callback(
                update_feature_groups_callback
            )

    @handle_errors
    @validate_input
    def fit(
        self,
        X: np.ndarray,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **fit_params: Dict[str, Any],
    ) -> "LongitudinalPipeline":
        """Fit the transformers in the pipeline and then the final estimator.

        For each step, the transformers are configured and fitted. The data is transformed
        and updated for each step, ensuring that the longitudinal feature structure is maintained.

        Args:
            X (np.ndarray):
                The input data.
            y (Optional[Union[pd.Series, np.ndarray]]):
                The target variable.
            **fit_params (Dict[str, Any]):
                Additional fitting parameters.

        Returns:
            LongitudinalPipeline:
                The fitted pipeline.

        """
        self._longitudinal_data = X.copy()
        self.selected_feature_indices_ = np.arange(X.shape[1])

        if y is not None:
            y = y.copy()

        filtered_steps = [(name, transformer) for name, transformer in self.steps[:-1] if transformer is not None]
        for step_idx, (name, transformer) in enumerate(filtered_steps):
            (
                transformer,
                self._longitudinal_data,
                self.selected_feature_indices_,
                self.feature_list_names,
            ) = configure_and_fit_transformer(
                transformer,
                name,
                self._longitudinal_data,
                y,
                fit_params,
                self.selected_feature_indices_,
                self.feature_list_names,
                self.features_group,
                self.non_longitudinal_features,
                self.update_feature_groups_callback,
            )
            (
                self._longitudinal_data,
                self.features_group,
                self.non_longitudinal_features,
                self.feature_list_names,
            ) = self._update_longitudinal_data_callback(name, step_idx, transformer, y)

        if self._final_estimator is not None:
            self._final_estimator = handle_final_estimator(
                self._final_estimator,
                self.steps,
                self.features_group,
                self.non_longitudinal_features,
                self.feature_list_names,
                self._longitudinal_data,
                y,
                fit_params,
            )

        return self

    def _update_longitudinal_data_callback(
        self, name: str, step_idx: int, transformer: TransformerMixin, y: Optional[Union[pd.Series, np.ndarray]]
    ) -> Tuple[np.ndarray, List[List[int]], List[Union[int, str]], List[str]]:
        """Update the longitudinal data and the features group using the update_data_callback function.

        Except for the final estimator, this function is commonly invoked for each pipeline transformer.
        This permits the longitudinal features to be updated throughout the process, rather than just at the end.
        This additional phase was required because, in the standard scikit-learn pipeline, data is not updated along
        the pipeline because it is unnecessary; we only require the final data (from the final transformer)
        to be passed to the final estimator. However, in this longitudinal-variant case we must also transmit the
        structure of the longitudinal features (features group) across pipeline's steps.

        Args:
            name (str):
                The name of the transformer.
            step_idx (int):
                The index of the transformer in the pipeline.
            transformer (TransformerMixin):
                The transformer.
            y (Optional[Union[pd.Series, np.ndarray]]):
                The target variable.

        Returns:
            Tuple[np.ndarray, List[List[int]], List[Union[int, str]], List[str]]: The updated longitudinal
            data, features group, non-longitudinal features, and feature list names.

        """
        df = pd.DataFrame(self._longitudinal_data, columns=self.feature_list_names)

        dummy_longitudinal_dataset = LongitudinalDataset(file_path=None, data_frame=df)
        dummy_longitudinal_dataset._feature_groups = self.features_group  # pylint: disable=W0212

        (
            updated_longitudinal_data,
            updated_features_group,
            updated_non_longitudinal_features,
            updated_feature_list_names,
        ) = self.update_feature_groups_callback(step_idx, dummy_longitudinal_dataset, y, name, transformer)

        return (
            updated_longitudinal_data,
            updated_features_group,
            updated_non_longitudinal_features,
            updated_feature_list_names,
        )

    @validate_input
    def predict(self, X: np.ndarray, **predict_params: Dict[str, Any]) -> np.ndarray:
        """Predict the target values using the final estimator of the pipeline.

        Args:
            X (np.ndarray):
                The input data.
            **predict_params (Dict[str, Any]):
                Additional prediction parameters.

        Returns:
            np.ndarray:
                Predicted values.

        Raises:
            NotImplementedError:
                If the final estimator does not have a predict method.

        """
        X = X[:, self.selected_feature_indices_]

        if hasattr(self._final_estimator, "predict"):
            return self._final_estimator.predict(X, **predict_params)
        raise NotImplementedError(f"predict is not implemented for this estimator: {type(self._final_estimator)}")

    @validate_input
    def predict_proba(self, X: np.ndarray, **predict_params: Dict[str, Any]) -> np.ndarray:
        X = X[:, self.selected_feature_indices_]

        if hasattr(self._final_estimator, "predict_proba"):
            return self._final_estimator.predict_proba(X, **predict_params)
        raise NotImplementedError(f"predict_proba is not implemented for this estimator: {type(self._final_estimator)}")

    @validate_input
    def transform(self, X: np.ndarray, **transform_params: Dict[str, Any]) -> np.ndarray:
        """Transform the input data using the final estimator of the pipeline.

        Args:
            X (np.ndarray):
                The input data.
            **transform_params (Dict[str, Any]):
                Additional transformation parameters.

        Returns:
            np.ndarray:
                Transformed data.

        """
        if self.selected_feature_indices_ is None or len(self.selected_feature_indices_) == 0:
            print("No feature selection was performed. Returning the original data.")
            return X
        X = X[:, self.selected_feature_indices_]
        return self._final_estimator.transform(X, **transform_params)

    @property
    def _final_estimator(self):
        return self.final_estimator

    @_final_estimator.setter
    def _final_estimator(self, value):
        self.final_estimator = value
