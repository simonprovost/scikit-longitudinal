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
    """Machine Learning-based Longitudinal Pipeline for handling and processing longitudinal techniques (preprocessors, classifier, etc.).

    The `LongitudinalPipeline` extends scikit-learn's `Pipeline` to provide specialised methods and attributes for working
    with longitudinal data. It ensures that the structure of longitudinal features is updated and maintained throughout
    the pipeline's transformations, making it ideal for longitudinal classification tasks.

    !!! question "Feature Groups and Non-Longitudinal Features"
        Two key attributes, `feature_groups` and `non_longitudinal_features`, enable algorithms to interpret the temporal
        structure of longitudinal data, we try to build those as much as possible for users, while allowing
        users to also define their own feature groups if needed. As follows:

        - **feature_groups**: A list of lists where each sublist contains indices of a longitudinal attribute's waves,
          ordered from oldest to most recent. This captures temporal dependencies.
        - **non_longitudinal_features**: A list of indices for static, non-temporal features excluded from the temporal
          matrix.

        Proper setup of these attributes is critical for leveraging temporal patterns effectively, and effectively
        use the primitives that follow.

        These attributes are updated dynamically as data passes through the pipeline, ensuring that temporal relationships
        are preserved.

        To see more, we highly recommend visiting the `Temporal Dependency` page in the documentation.
        [Temporal Dependency Guide :fontawesome-solid-timeline:](https://scikit-longitudinal.readthedocs.io/latest/tutorials/temporal_dependency/){ .md-button }

    !!! note "Extension of scikit-learn's Pipeline"
        While maintaining the interface of scikit-learn's `Pipeline`, this class includes additional validations and
        methods to ensure the correct processing of longitudinal data. It integrates seamlessly with scikit-learn's
        ecosystem, allowing for the use of standard transformers and estimators as well.

        No need to keep it `Sklong` only, you can use any scikit-learn compatible transformer or estimator.

    Args:
        steps (List[Tuple[str, Any]]): List of (name, transform) tuples that are chained in the order they are provided.
            The last object should be an estimator.
        features_group (List[List[int]]): A temporal matrix where each sublist contains indices of a longitudinal
            attribute's waves.
        non_longitudinal_features (List[Union[int, str]], optional): List of indices or names of non-longitudinal
            features. Defaults to None.
        update_feature_groups_callback (Union[Callable, str], optional): Callback function to update feature groups
            during transformations. Can be a string for built-in callbacks or a custom function. Defaults to None.
        feature_list_names (List[str], optional): List of feature names corresponding to the dataset columns. Defaults
            to None.

    Attributes:
        _longitudinal_data (np.ndarray): The longitudinal data being processed.
        selected_feature_indices_ (np.ndarray): Indices of the selected features after transformations.
        final_estimator (Any): The final estimator in the pipeline.


    ??? question "What is all about with Custom Callback Function?"
        The `update_feature_groups_callback` parameter allows users to customise how feature groups and non-longitudinal
        features are updated after each transformation in the pipeline. This is crucial for maintaining the temporal
        structure of longitudinal data as it passes through various preprocessing steps.

        What should I put when I am not sure — What'ss the default? literally, `"default"`. We cover it up for you, but you can also define your own logic to handle specific cases or
        transformations that may alter the structure of the data. This flexibility is particularly useful when dealing
        with complex datasets or when using custom transformers that may not conform to the standard behaviour expected
        by the pipeline.

        In a nutshell:

        - [x] **Dynamic Updates**: The callback ensures that `features_group` and `non_longitudinal_features` are updated after
          each transformation, preserving the temporal relationships in the data.
        - [x] **Flexibility**: It provides a mechanism for users to inject custom logic tailored to their specific dataset or
          preprocessing needs.

        #### Custom Implementation
        - Users can define their own callback function to handle specialised requirements. The function must follow this signature:
            ```python
            def callback(
                step_idx: int,
                dummy_longitudinal_dataset: LongitudinalDataset,
                y: Union[pd.Series, np.ndarray],
                name: str,
                transformer: TransformerMixin
            ) -> Tuple[np.ndarray, List[List[int]], List[Union[int, str]], List[str]]:
                # Custom logic here
                pass
            ```

        - **Parameters**:

              - `step_idx`: The index of the current step in the pipeline.
              - `dummy_longitudinal_dataset`: A `LongitudinalDataset` instance representing the current state of the data.
              - `y`: The target variable (if provided).
              - `name`: The name of the current transformer.
              - `transformer`: The transformer being applied at this step.

        - **Returns**: A tuple containing:

              - Updated longitudinal data (`np.ndarray`).
              - Updated feature groups (`List[List[int]]`).
              - Updated non-longitudinal features (`List[Union[int, str]]`).
              - Updated feature names (`List[str]`).

        #### Usage Example

        - You can pass a custom function or even a lambda for quick adjustments:

          ```python
          def custom_callback(step_idx, dataset, y, name, transformer):
              updated_data = transformer.transform(dataset.data)
              updated_groups = dataset.feature_groups()  # Custom logic can modify this
              updated_non_long = dataset.non_longitudinal_features()
              updated_names = dataset.data.columns.tolist()
              return updated_data, updated_groups, updated_non_long, updated_names

          pipeline = LongitudinalPipeline(
              steps=[('transformer', SomeTransformer()), ('classifier', SomeClassifier())],
              features_group=[[0, 1, 2], [3, 4, 5]],
              update_feature_groups_callback=custom_callback
          )
          ```

        - Or use a lambda for simplicity:

          ```python
          pipeline = LongitudinalPipeline(
              steps=[...],
              features_group=[...],
              update_feature_groups_callback=lambda step_idx, dataset, y, name, transformer: (
                  transformer.transform(dataset.data),
                  dataset.feature_groups(),
                  dataset.non_longitudinal_features(),
                  dataset.data.columns.tolist()
              )
          )
          ```


    Examples:
        Below are examples demonstrating the usage of the `LongitudinalPipeline` class.

        !!! example "Basic Usage with a Classifier"
            ```python
            from scikit_longitudinal.pipeline import LongitudinalPipeline
            from scikit_longitudinal.data_preparation import LongitudinalDataset
            from scikit_longitudinal.estimators.trees import LexicoDecisionTreeClassifier
            from scikit_longitudinal.data_preparation import LongitudinalDataset
            from scikit_longitudinal.data_preparation import MerWavTimePlus

            # Load dataset
            dataset = LongitudinalDataset('./stroke_longitudinal.csv')
            dataset.load_data()
            dataset.load_target(target_column="stroke_w2")
            dataset.setup_features_group("elsa")
            dataset.load_train_test_split(test_size=0.2, random_state=42)

            # Define pipeline steps with LexicoDecisionTreeClassifier
            steps = [
                ('MerWavTime Plus', MerWavTimePlus()), # Recall, a pipeline is at least two steps and the first one being a Data Transformation step. Here as we use a Longitudinal classifier, we need to use MerWavTimePlus, retaining the temporal dependency.
                # Feel free to add more steps like a feature selection step.
                ('classifier', LexicoDecisionTreeClassifier(features_group=dataset.feature_groups()))
            ]

            # Note if you would like to do a pipeline of non-longitudinal classifier like RandomForestClassifier,
            # rather than LexicoRandomForestClassifier, you can always use `Sklearn` pipeline directly, as follows:
            # from sklearn.ensemble import RandomForestClassifier
            # steps = [
            #     ('AggrFunc', AggrFunc()),
            #     ('classifier', RandomForestClassifier())
            # ]

            # Initialize pipeline
            pipeline = LongitudinalPipeline(
                steps=steps,
                features_group=dataset.feature_groups(),
                non_longitudinal_features=dataset.non_longitudinal_features(),
                feature_list_names=dataset.data.columns.tolist(),
                update_feature_groups_callback="default"
            )

            # Fit and predict
            pipeline.fit(dataset.X_train, dataset.y_train)
            y_pred = pipeline.predict(dataset.X_test)
            print(f"Predictions: {y_pred}")
            ```

        !!! example "Using a Custom Callback"
            ```python
            from scikit_longitudinal.pipeline import LongitudinalPipeline

            # Define a custom callback function
            def custom_callback(step_idx, dataset, y, name, transformer):
                # Custom logic to update feature groups
                updated_data = transformer.transform(dataset.data)
                updated_groups = dataset.feature_groups()
                updated_non_long = dataset.non_longitudinal_features()
                updated_names = dataset.data.columns.tolist()
                return updated_data, updated_groups, updated_non_long, updated_names

            # Initialize pipeline with custom callback
            pipeline = LongitudinalPipeline(
                steps=[...],
                features_group=[...],
                update_feature_groups_callback=custom_callback
            )
            ```
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
        """Fit the transformers and the final estimator in the pipeline.

        This method iterates through each transformer in the pipeline, configuring and fitting them while updating the
        longitudinal data and feature groups. The final estimator is then fitted using the transformed data.

        Args:
            X (np.ndarray): Input data.
            y (Optional[Union[pd.Series, np.ndarray]]): Target variable.
            **fit_params (Dict[str, Any]): Additional fitting parameters.

        Returns:
            LongitudinalPipeline: The fitted pipeline.
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
        """Update the longitudinal data and feature groups using the callback function.

        This method is called after each transformer to update the longitudinal data and feature groups, ensuring that
        the temporal structure is maintained throughout the pipeline.

        Args:
            name (str): Name of the transformer.
            step_idx (int): Index of the transformer in the pipeline.
            transformer (TransformerMixin): The transformer.
            y (Optional[Union[pd.Series, np.ndarray]]): Target variable.

        Returns:
            Tuple[np.ndarray, List[List[int]], List[Union[int, str]], List[str]]: Updated longitudinal data, feature
                groups, non-longitudinal features, and feature list names.
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
        """Predict target values using the final estimator.

        Applies the selected feature indices to the input data and uses the final estimator to make predictions.

        Args:
            X (np.ndarray): Input data.
            **predict_params (Dict[str, Any]): Additional prediction parameters.

        Returns:
            np.ndarray: Predicted values.

        Raises:
            NotImplementedError: If the final estimator does not implement `predict`.
        """
        X = X[:, self.selected_feature_indices_]

        if hasattr(self._final_estimator, "predict"):
            return self._final_estimator.predict(X, **predict_params)
        raise NotImplementedError(f"predict is not implemented for this estimator: {type(self._final_estimator)}")

    @validate_input
    def predict_proba(self, X: np.ndarray, **predict_params: Dict[str, Any]) -> np.ndarray:
        """Predict class probabilities using the final estimator.

        Applies the selected feature indices to the input data and uses the final estimator to predict probabilities.

        Args:
            X (np.ndarray): Input data.
            **predict_params (Dict[str, Any]): Additional prediction parameters.

        Returns:
            np.ndarray: Predicted probabilities.

        Raises:
            NotImplementedError: If the final estimator does not implement `predict_proba`.
        """
        X = X[:, self.selected_feature_indices_]

        if hasattr(self._final_estimator, "predict_proba"):
            return self._final_estimator.predict_proba(X, **predict_params)
        raise NotImplementedError(f"predict_proba is not implemented for this estimator: {type(self._final_estimator)}")

    @validate_input
    def transform(self, X: np.ndarray, **transform_params: Dict[str, Any]) -> np.ndarray:
        """Transform the input data using the final estimator.

        Applies the selected feature indices and transforms the data using the final estimator's `transform` method.

        Args:
            X (np.ndarray): Input data.
            **transform_params (Dict[str, Any]): Additional transformation parameters.

        Returns:
            np.ndarray: Transformed data.
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
