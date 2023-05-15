from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # pragma: no cover

import numpy as np  # pragma: no cover
import pandas as pd  # pragma: no cover
from sklearn.pipeline import Pipeline  # pragma: no cover

from scikit_longitudinal.data_preparation import LongitudinalDataset  # pragma: no cover


# pylint: disable=W0511
class LongitudinalPipeline(Pipeline):  # pragma: no cover
    """A custom scikit-learn pipeline designed for longitudinal based machine learning algorithms.

    The LongitudinalPipeline works similarly to the standard scikit-learn
    pipeline but specifically handles longitudinal data. Thus, it internally keeps track of
    the longitudinal dataset with access to grouping features to pass an
    updated version to any step in need. It also manages train-test data and
    target shapes, updating them along the process. As a result, the predict method
    does not require any data to be passed as a parameter as the internal's one will be used.

    Attributes:
        longitudinal_data (LongitudinalDataset): The longitudinal dataset object containing the input data.

    """

    def __init__(self, steps: List[Tuple[str, Any]]) -> None:
        """Initialize the LongitudinalPipeline.

        Args:
            steps (list)
                List of (name, transform) tuples specifying the sequence of transforms and the final estimator.

        """
        super().__init__(steps=steps)
        self._longitudinal_data: Optional["LongitudinalDataset"] = None

    @property
    def longitudinal_data(self) -> "LongitudinalDataset":
        """Get the longitudinal dataset object containing the input data.

        Returns:
            LongitudinalDataset:
                The longitudinal dataset object containing the input data.

        """
        return self._longitudinal_data

    def fit(
        self,
        X: "LongitudinalDataset",
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        update_data_callback: Optional[Callable] = None,
        **fit_params: Dict,
    ) -> "LongitudinalPipeline":
        """Fit the model by executing the pipeline steps.

        Args:
            X (LongitudinalDataset):
                The longitudinal dataset object containing the input data.
            y (Union[pd.Series, np.ndarray, None], optional):
                The target data. If not provided, the method will be called without the target data. Defaults to None.
            update_data_callback (Callable, optional):
                An optional callback function to update the dataset at each step. Defaults to None.
            fit_params (dict, optional):
                Additional fit parameters. Defaults to None.

        Returns:
            LongitudinalPipeline: This estimator.

        """
        if not isinstance(X, LongitudinalDataset):
            raise ValueError("X must be a LongitudinalDataset object")
        if X.data.empty:
            raise ValueError("X must have a data attribute")

        self._longitudinal_data = X
        if y is not None:
            y = y.copy()

        filtered_steps = [(name, transformer) for name, transformer in self.steps[:-1] if transformer is not None]
        for step_idx, (name, transformer) in enumerate(filtered_steps):
            if hasattr(transformer, "features_group"):
                transformer.features_group = self._longitudinal_data.feature_groups()

            X_transformed = transformer.fit_transform(self._longitudinal_data.X_train, self._longitudinal_data.y_train)
            self._longitudinal_data.X_train.iloc[:, :] = X_transformed

            if update_data_callback is not None:
                updated_longitudinal_data = update_data_callback(
                    self, step_idx, self._longitudinal_data, y, name, transformer
                )
                if updated_longitudinal_data is not None:
                    self._longitudinal_data = updated_longitudinal_data
                    # TODO Improve the parameters to be passed here
                    self._longitudinal_data.setup_features_group("elsa")
                    if hasattr(transformer, "selected_features_"):
                        self._longitudinal_data.setX_train(
                            self._longitudinal_data.X_train.iloc[:, transformer.selected_features_]
                        )
                        self._longitudinal_data.setX_test(
                            self._longitudinal_data.X_test.iloc[:, transformer.selected_features_]
                        )

        if self._final_estimator is not None:
            if hasattr(self._final_estimator, "features_group"):
                self._final_estimator.features_group = self._longitudinal_data.feature_groups()
                print(self._longitudinal_data.feature_groups(names=True))
            self._final_estimator.fit(self._longitudinal_data.X_train, self._longitudinal_data.y_train)
        return self

    def predict(self, X: Optional[Any] = None, **predict_params: Dict) -> np.ndarray:
        """Predict the target for the test data using the final estimator of the pipeline.

        Args:
            X (ignored):
                This parameter is ignored as the method uses the X_test attribute of the class.
            predict_params (dict, optional):
                Additional parameters for the predict method of the final estimator. Defaults to None.

        Returns:
            np.ndarray: The predicted target values.

        """
        if self._final_estimator is None:
            raise ValueError("No final estimator is set in the pipeline.")

        if not hasattr(self._final_estimator, "predict"):
            raise ValueError("The final estimator does not have a predict method.")

        if self._longitudinal_data.X_test is None:
            raise ValueError("X_test attribute is not set. Make sure the pipeline has been fitted.")

        return self._final_estimator.predict(self._longitudinal_data.X_test, **predict_params)

    def transform(self, X: Optional[Any] = None, **transform_params: Dict) -> np.ndarray:
        """Transform the test data using the final estimator of the pipeline.

        Args:
            X (ignored):
                This parameter is ignored as the method uses the X_test attribute of the class.
            transform_params (dict, optional):
                Additional parameters for the transform method of the final estimator. Defaults to None.

        Returns:
            np.ndarray: The transformed test data.

        """
        if self._final_estimator is None:
            raise ValueError("No final estimator is set in the pipeline.")

        if not hasattr(self._final_estimator, "transform"):
            raise ValueError("The final estimator does not have a transform method.")

        if self._longitudinal_data.X_test is None:
            raise ValueError("X_test attribute is not set. Make sure the pipeline has been fitted.")

        return self._final_estimator.transform(self._longitudinal_data.X_test, **transform_params)

    def get_dataset(self) -> "LongitudinalDataset":
        """Retrieve the current longitudinal dataset from the pipeline.

        Returns:
            LongitudinalDataset:
                The current dataset in the pipeline.

        """
        return self._longitudinal_data
