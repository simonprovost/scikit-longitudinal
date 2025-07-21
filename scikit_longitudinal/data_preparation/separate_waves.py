# pylint: disable=R0801

from typing import List, Union

import numpy as np
import pandas as pd
import ray
from overrides import override
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from scikit_longitudinal.data_preparation.longitudinal_dataset import clean_padding
from scikit_longitudinal.estimators.ensemble.longitudinal_stacking.longitudinal_stacking import (
    LongitudinalStackingClassifier,
)
from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import (
    LongitudinalEnsemblingStrategy,
    LongitudinalVotingClassifier,
)
from scikit_longitudinal.templates import CustomClassifierMixinEstimator
from scikit_longitudinal.templates.custom_data_preparation_mixin import DataPreparationMixin


def validate_extract_wave_input(func):
    """Decorator to validate the input to the _extract_wave function.

    Args:
        func (function):
            Function to be decorated.

    Returns:
        function:
            Decorated function.

    Raises:
        ValueError: If the wave number is less than 0.

    """

    def wrapper(self, wave: int, extract_indices: bool = False):
        if not isinstance(extract_indices, bool):
            raise TypeError(f"Invalid type for extract_indices: {type(extract_indices)}. It should be a boolean.")
        if wave < 0:
            raise ValueError(f"Invalid wave number: {wave}. It should be more than 0")
        return func(self, wave, extract_indices)

    return wrapper


def validate_extract_wave_output(func):  # pragma: no cover
    """Decorator to validate the output of the _extract_wave function.

    Args:
        func (function):
            Function to be decorated.

    Returns:
        function:
            Decorated function.

    Raises:
        ValueError: If the number of features in the wave does not match the expected number of features.

    """

    def wrapper(self, wave: int, extract_indices: bool = False):
        if extract_indices:
            X_wave, y_wave, extracted_indices = func(self, wave, extract_indices)
        else:
            X_wave, y_wave = func(self, wave, extract_indices)
        expected_features = len([group[wave] for group in self.features_group if wave < len(group)]) + len(
            self.non_longitudinal_features
        )
        if X_wave.shape[1] != expected_features:
            raise ValueError(f"Invalid number of features in X_wave: {X_wave.shape[1]}. Expected {expected_features}.")
        if extract_indices:
            return X_wave, y_wave, extracted_indices
        return X_wave, y_wave

    return wrapper


def validate_fit_input(func):
    """Decorator to validate the input to the fit function.

    Args:
        func (function):
            Function to be decorated.

    Returns:
        function: Decorated function.

    Raises:
        ValueError: If the classifier, dataset, or feature groups are None.

    """

    def wrapper(self, X, y):
        if self.estimator is None or self.dataset is None or self.features_group is None:
            raise ValueError("The classifier, dataset, and feature groups must not be None.")
        return func(self, X, y)

    return wrapper


def validate_fit_output(func):  # pragma: no cover
    """Decorator to validate the output of the fit function.

    Args:
        func (function): Function to be decorated.

    Returns:
        function:
            Decorated function.

    Raises:
        ValueError: If the clf_ensemble is None after fitting.

    """

    def wrapper(self, X, y):
        result = func(self, X, y)
        if self.clf_ensemble is None:
            raise ValueError("Fit failed: clf_ensemble is None.")
        return result

    return wrapper


def validate_predict_input(func):
    """Decorator to validate the input to the predict function.

    Args:
        func (function):
            Function to be decorated.

    Returns:
        function:
            Decorated function.

    Raises:
        NotFittedError: If the SepWav instance is not fitted yet.

    """

    def wrapper(self, X):
        if self.clf_ensemble is None:
            raise NotFittedError(
                "This SepWav instance is not fitted yet. Call 'fit' with appropriate arguments "
                "before using this estimator."
            )
        return func(self, X)

    return wrapper


def validate_predict_wave_input(func):
    """Decorator to validate the input to the predict wave function.

    Args:
        func (function):
            Function to be decorated.

    Returns:
        function:
            Decorated function.

    Raises:
        NotFittedError: If the SepWav instance is not fitted yet.

    """

    def wrapper(self, wave, X):
        if self.estimators is None or len(self.estimators) == 0:
            raise NotFittedError(
                "This SepWav instance is not fitted yet. Call 'fit' with appropriate arguments "
                "before using this estimator."
            )
        if not 0 <= wave < len(self.estimators):
            raise IndexError(f"Invalid wave number: {wave}.")
        return func(self, wave, X)

    return wrapper


@ray.remote
def train_classifier(classifier, X_wave, y_wave, wave):  # pragma: no cover
    """Train a classifier on a specific wave.

    This function is used for parallel processing by leveraging Ray framework.

    Args:
        classifier (BaseEstimator):
            The classifier to employ.
        X_wave (DataFrame):
            The input samples for the wave. Each row represents an observation,
            and each column represents a feature.
        y_wave (Series):
            The target values for the wave.
        wave (int):
            The wave number to use as for the returned classifier by denoting it as wave_{wave number}.

    Returns:
        tuple: A tuple containing two elements:
            - wave (string): The wave number as a string.
            - clf_wave (BaseEstimator): The trained classifier.

    """
    clf_wave = clone(classifier)
    if hasattr(X_wave, "values") and hasattr(y_wave, "values"):
        X_wave = X_wave.values
        y_wave = y_wave.values
    clf_wave.fit(X_wave, y_wave)
    return f"wave_{wave}", clf_wave


# pylint: disable=too-many-instance-attributes, too-many-arguments
class SepWav(BaseEstimator, ClassifierMixin, DataPreparationMixin):
    """SepWav stands for Separate Waves, a training done wave-by-wave for longitudinal dataset.

    The `SepWav` class implements the Separate Waves strategy, treating each wave (time point) as a separate dataset.
    A classifier is trained on each wave independently, and their predictions are combined using ensemble methods
    such as voting or stacking.

    !!! question "What is a feature group?"
        In a nutshell, a feature group is a collection of features sharing a common base longitudinal attribute
        across different waves of data collection (e.g., "income_wave1", "income_wave2", "income_wave3").

        To see more, we highly recommend visiting the `Temporal Dependency` page in the documentation.

        [Temporal Dependency Guide :fontawesome-solid-timeline:](https://scikit-longitudinal.readthedocs.io/latest/tutorials/temporal_dependency/){ .md-button }

    !!! note "Ensemble Strategies"
        Supported ensemble methods include:

        - [x] Simple majority voting
        - [x] Weighted voting (e.g., decaying weights for older waves)
        - [x] Stacking with a meta-learner

        Refer to `LongitudinalVoting` and `LongitudinalStacking` for mathematical details.

    Args:
        estimator (Union[ClassifierMixin, CustomClassifierMixinEstimator], optional): Base classifier for each wave. Defaults to None.
        features_group (List[List[int]], optional): Temporal matrix where each sublist contains indices of a longitudinal attribute's waves. Defaults to None.
        non_longitudinal_features (List[Union[int, str]], optional): List of indices or names of non-longitudinal features. Defaults to None.
        feature_list_names (List[str], optional): List of feature names in the dataset. Defaults to None.
        voting (LongitudinalEnsemblingStrategy, optional): Ensemble strategy. Defaults to `LongitudinalEnsemblingStrategy.MAJORITY_VOTING`.
        stacking_meta_learner (Union[CustomClassifierMixinEstimator, ClassifierMixin, None], optional): Meta-learner for stacking. Defaults to `LogisticRegression()`.
        n_jobs (int, optional): Number of parallel jobs. Defaults to None.
        parallel (bool, optional): Whether to run wave fitting in parallel. Defaults to False.
        num_cpus (int, optional): Number of CPUs for parallel processing. Defaults to -1 (all available CPUs).

    Attributes:
        dataset (pd.DataFrame): Training dataset.
        estimator (BaseEstimator): Base classifier for each wave.
        estimators (List): List of trained classifiers for each wave.
        voting (LongitudinalEnsemblingStrategy): Ensemble strategy used.
        stacking_meta_learner (Union[CustomClassifierMixinEstimator, ClassifierMixin]): Meta-learner for stacking.
        clf_ensemble (BaseEstimator): Combined ensemble classifier.
        n_jobs (int): Number of parallel jobs.
        parallel (bool): Whether parallel processing is enabled.
        num_cpus (int): Number of CPUs used.

    Examples:
        Below are examples using the "stroke.csv" dataset. Replace "stroke.csv" with your actual dataset path.

        !!! example "Basic Usage with Majority Voting"
            ```python
            from scikit_longitudinal.data_preparation import LongitudinalDataset
            from scikit_longitudinal.data_preparation import SepWav
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import LongitudinalEnsemblingStrategy

            # Load dataset
            dataset = LongitudinalDataset('./stroke_longitudinal.csv')
            dataset.load_data()
            dataset.load_target(target_column="stroke_w2")
            dataset.setup_features_group("elsa")
            dataset.load_train_test_split(test_size=0.2, random_state=42)

            # Initialize classifier
            classifier = RandomForestClassifier()

            # Initialize SepWav
            sepwav = SepWav(
                estimator=classifier,
                features_group=dataset.feature_groups(),
                non_longitudinal_features=dataset.non_longitudinal_features(),
                feature_list_names=dataset.data.columns.tolist(),
                voting=LongitudinalEnsemblingStrategy.MAJORITY_VOTING
            )

            # Fit and predict
            sepwav.fit(dataset.X_train, dataset.y_train)
            y_pred = sepwav.predict(dataset.X_test)

            # Evaluate
            accuracy = accuracy_score(dataset.y_test, y_pred)
            print(f"Accuracy: {accuracy}")
            ```

        !!! example "Using Stacking Ensemble"
            ```python
            from scikit_longitudinal.data_preparation import LongitudinalDataset
            from scikit_longitudinal.data_preparation import SepWav
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.linear_model import LogisticRegression
            from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import LongitudinalEnsemblingStrategy


            # Load dataset
            dataset = LongitudinalDataset('./stroke_longitudinal.csv')
            dataset.load_data()
            dataset.load_target(target_column="stroke_w2")
            dataset.setup_features_group("elsa")
            dataset.load_train_test_split(test_size=0.2, random_state=42)

            # Initialize classifier
            classifier = RandomForestClassifier()

            # Initialize SepWav with stacking
            sepwav = SepWav(
                estimator=classifier,
                features_group=dataset.feature_groups(),
                non_longitudinal_features=dataset.non_longitudinal_features(),
                feature_list_names=dataset.data.columns.tolist(),
                voting=LongitudinalEnsemblingStrategy.STACKING,
                stacking_meta_learner=LogisticRegression()
            )

            # Fit and predict
            sepwav.fit(dataset.X_train, dataset.y_train)
            y_pred = sepwav.predict(dataset.X_test)

            # Evaluate
            accuracy = accuracy_score(dataset.y_test, y_pred)
            print(f"Accuracy: {accuracy}")
            ```

        !!! example "Using Parallel Processing"
            ```python
            # ... Similar to the previous example, but with parallel processing enabled ...

            # Initialize SepWav with parallel processing
            sepwav = SepWav(
                estimator=classifier,
                features_group=dataset.feature_groups(),
                non_longitudinal_features=dataset.non_longitudinal_features(),
                feature_list_names=dataset.data.columns.tolist(),
                parallel=True, # Enable parallel processing
                num_cpus=4 # Specify number of CPUs to use (or -1 for all available CPUs)
            )

            # ... Similar to the previous example, but with parallel processing enabled ...
            ```
    """

    def __init__(
        self,
        estimator: Union[ClassifierMixin, CustomClassifierMixinEstimator] = None,
        features_group: List[List[int]] = None,
        non_longitudinal_features: List[Union[int, str]] = None,
        feature_list_names: List[str] = None,
        voting: LongitudinalEnsemblingStrategy = LongitudinalEnsemblingStrategy.MAJORITY_VOTING,
        stacking_meta_learner: Union[CustomClassifierMixinEstimator, ClassifierMixin, None] = LogisticRegression(),
        n_jobs: int = None,
        parallel: bool = False,
        num_cpus: int = -1,
    ):
        self.features_group = features_group
        self.non_longitudinal_features = non_longitudinal_features
        self.feature_list_names = feature_list_names

        self.estimator = estimator
        self.voting = voting
        self.stacking_meta_learner = stacking_meta_learner

        self.n_jobs = n_jobs
        self.parallel = parallel
        self.num_cpus = num_cpus

        self.estimators = []
        self.dataset = pd.DataFrame([])
        self.target = np.ndarray([])
        self.clf_ensemble = None

        if self.parallel and ray.is_initialized() is False:  # pragma: no cover
            if self.num_cpus != -1:
                ray.init(num_cpus=self.num_cpus)
            else:
                ray.init()

    @override
    def _prepare_data(self, X: np.ndarray, y: np.ndarray = None) -> "SepWav":
        """Prepare the data for transformation.

        In `SepWav`, data preparation is handled within the `fit` method. This method is overridden for compatibility
        with `DataPreparationMixin` but performs no operations.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray, optional): Target data. Defaults to None.

        Returns:
            SepWav: The instance itself.
        """
        return self

    @property
    def classes_(self):
        if self.clf_ensemble is None:
            raise NotFittedError("This SepWav instance is not fitted yet. Call 'fit' with appropriate arguments.")
        return self.clf_ensemble.classes_

    @validate_extract_wave_input
    @validate_extract_wave_output
    def _extract_wave(self, wave: int, extract_indices: bool = False) -> Union[pd.DataFrame, pd.Series, list]:
        """Extract a specific wave from the dataset for training.

        Args:
            wave (int): Wave number to extract (0-based index).
            extract_indices (bool, optional): Whether to return feature indices. Defaults to False.

        Returns:
            tuple: If extract_indices is True, returns (X_wave, y_wave, feature_indices); otherwise, (X_wave, y_wave).

                - [x] X_wave (pd.DataFrame): Input samples for the wave.
                - [x] y_wave (pd.Series): Target values for the wave.
                - [x] feature_indices (list): Indices of extracted features (if extract_indices is True).

        Raises:
            ValueError: If wave number is negative.
        """
        feature_indices = [group[wave] for group in self.features_group if wave < len(group)]
        if self.non_longitudinal_features is not None:
            feature_indices.extend(self.non_longitudinal_features)

        X_wave = self.dataset.iloc[:, feature_indices]
        y_wave = self.target

        if extract_indices and feature_indices:
            return X_wave, y_wave, feature_indices
        return X_wave, y_wave

    # pylint: disable=unused-argument
    @validate_fit_input
    @validate_fit_output
    def fit(self, X: Union[List[List[float]], "np.ndarray"], y: Union[List[float], "np.ndarray"]):
        """Fit the SepWav model to the training data.

        Trains a classifier for each wave and combines them using the specified ensemble strategy.

        Args:
            X (Union[List[List[float]], np.ndarray]): Input samples.
            y (Union[List[float], np.ndarray]): Target values.

        Returns:
            SepWav: Fitted instance.

        Raises:
            ValueError: If required parameters (estimator, features_group) are None or ensemble strategy is invalid.
        """
        self.dataset = pd.DataFrame(X, columns=self.feature_list_names)
        self.target = y

        if self.features_group is not None:
            self.features_group = clean_padding(self.features_group)

        if self.parallel:
            futures = [
                train_classifier.remote(self.estimator, X_train, y_train, wave)
                for wave, (X_train, y_train) in enumerate(
                    self._extract_wave(wave=i) for i in range(max(len(group) for group in self.features_group))
                )
            ]
            self.estimators = ray.get(futures)
        else:
            for i in range(max(len(group) for group in self.features_group)):
                X_wave, y_wave = self._extract_wave(wave=i)
                clf_wave = clone(self.estimator)
                if hasattr(X_wave, "values") and hasattr(y_wave, "values"):
                    X_wave = X_wave.values
                    y_wave = y_wave.values
                clf_wave.fit(X_wave, y_wave)
                self.estimators.append((f"wave_{i}", clf_wave))

        if self.voting == LongitudinalEnsemblingStrategy.STACKING:
            self.clf_ensemble = LongitudinalStackingClassifier(
                estimators=self.estimators, meta_learner=self.stacking_meta_learner, n_jobs=self.n_jobs
            )
        else:
            self.clf_ensemble = LongitudinalVotingClassifier(
                estimators=self.estimators,
                voting=self.voting,
                extract_wave=self._extract_wave,
                n_jobs=self.n_jobs,
            )

        X_data = self.dataset.values

        if hasattr(X_data, "flags") and not X_data.flags["C_CONTIGUOUS"]:
            X_data = np.ascontiguousarray(X_data)

        self.clf_ensemble.fit(X_data, self.target)

        return self

    @validate_predict_input
    def predict(self, X: Union[List[List[float]], "np.ndarray"]) -> Union[List[float], "np.ndarray"]:
        """Predict class labels for input samples.

        Uses the ensemble classifier to combine predictions from individual wave classifiers.

        Args:
            X (Union[List[List[float]], np.ndarray]): Input samples.

        Returns:
            Union[List[float], np.ndarray]: Predicted class labels.

        Raises:
            NotImplementedError: If the ensemble classifier does not support prediction.
        """
        if hasattr(self.clf_ensemble, "predict"):
            return self.clf_ensemble.predict(X)
        raise NotImplementedError(
            f"predict is not implemented for this classifier: {self.clf_ensemble} / type: {type(self.clf_ensemble)}"
        )

    @validate_predict_input
    def predict_proba(self, X: Union[List[List[float]], "np.ndarray"]) -> Union[List[List[float]], "np.ndarray"]:
        """Predict class probabilities for input samples.

        Computes probabilities using the ensemble classifier's `predict_proba` method, if available.

        Args:
            X (Union[List[List[float]], np.ndarray]): Input samples.

        Returns:
            Union[List[List[float]], np.ndarray]: Predicted class probabilities.

        Raises:
            NotImplementedError: If the ensemble classifier does not support probability predictions.
        """
        if hasattr(self.clf_ensemble, "predict_proba"):
            return self.clf_ensemble.predict_proba(X)
        raise NotImplementedError(
            "predict_proba is not implemented for this classifier: "
            f"{self.clf_ensemble} / type: {type(self.clf_ensemble)}"
        )

    @validate_predict_wave_input
    def predict_wave(self, wave: int, X: Union[List[List[float]], "np.ndarray"]) -> Union[List[float], "np.ndarray"]:
        """Predict class labels using the classifier for a specific wave.

        Useful for analyzing wave-specific performance or custom ensemble strategies.

        Args:
            wave (int): Wave number (0-based index).
            X (Union[List[List[float]], np.ndarray]): Input samples.

        Returns:
            Union[List[float], np.ndarray]: Predicted class labels for the specified wave.
        """
        return self.estimators[wave][1].predict(X)
