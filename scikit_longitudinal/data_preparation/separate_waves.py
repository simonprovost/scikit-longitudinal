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
    """The SepWav (Separate Waves) class for data transformation in longitudinal data analysis.

    ⚠️ Scikit-Longitudinal's docstrings will be updated to reflect the most recent documentation available on Github.
    If something is inconsistent, consult the documentation first, then file an issue. ⚠️

    The proposed methodology entails the provision of a user-defined algorithm in a sequential manner, wave by wave.
    The algorithm is designed to predict a class variable by being trained on a dataset that exclusively consists of
    the current wave, starting from the first wave. The algorithm undergoes training for each subsequent wave by
    utilising a freshly generated dataset that incorporates the newly introduced wave. The execution of this code
    snippet results in the generation of multiple classifiers. Next, it is possible to merge these classifiers into
    an ensemble model or employ them in alternative manners according to methods defined by the user. When the
    Predict function is called, the ensemble model will be utilised to make predictions on the class variable. To
    facilitate the prediction of the class variable for a specific wave, the user can utilise the predict_wave
    method. Alternatively, users can incorporate the list of classifiers into their script and manipulate them
    accordingly. The supported ensemble strategies encompass longitudinal voting and longitudinal stacking
    methodologies.

    Attributes:
        dataset (LongitudinalDataset):
            The dataset to be used.
        classifier (BaseEstimator):
            The classifier to be used.
        classifiers (List):
            A list of classifiers for each wave.
        voting (LongitudinalEnsemblingStrategy):
            The ensemble strategy to be used. The supported strategies are available in the EnsemblingStrategy enum.
        stacking_meta_learner (Union[CustomClassifierMixinEstimator, ClassifierMixin]):
            The final estimator to be used in stacking.
        clf_ensemble (BaseEstimator):
            The combined classifier.
        n_jobs (int):
            The number of jobs to run in parallel.
        parallel (bool):
            Whether to run the fit waves in parallel.
        num_cpus (int):
            The number of cpus to use for parallel processing.

    Example:
        ```python
        from sklearn.ensemble import RandomForestClassifier
        from scikit_longitudinal.data_preparation import LongitudinalDataset

        # Initialize the longitudinal data
        longitudinal_data = LongitudinalDataset("path/to/dataset.csv")
        longitudinal_data.load_data_target_train_test_split(
            target_column="target_column_name",
            remove_target_waves=True,
        )
        longitudinal_data.setup_features_group(input_data="elsa")

        # Initialise the classifier
        classifier = RandomForestClassifier()

        # Initialize the SepWav instance
        sepwav = SepWav(
            estimator=classifier,
            features_group=longitudinal_data.feature_groups(),
            non_longitudinal_features=longitudinal_data.non_longitudinal_features(),
            feature_list_names=longitudinal_data.data.columns.tolist(),
            voting=LongitudinalEnsemblingStrategy.MAJORITY_VOTING
        )

        # Fit and predict
        sepwav.fit(longitudinal_data.X_train, longitudinal_data.y_train)
        y_pred = sepwav.predict(longitudinal_data.X_test)
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
        """Prepare the data for the transformation.

        Replaced by the fit method.

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
        """Extracts a specific wave from the dataset for training.

        This method uses the wave number to select specific features from the dataset for training.
        It identifies the features that correspond to the wave, extends it with non-longitudinal features,
        and slices the dataset to create the training data for the wave.

        Args:
            wave (int):
                The wave number to extract. This should be a non-negative integer.
            extract_indices (bool):
                Whether to return the indices of the features that were extracted.

        Returns:
            A set of three elements is returned if extract_indices is True:
                - X_wave (DataFrame): The input samples for the wave. Each row represents an observation,
                    and each column represents a feature.
                - y_wave (Series): The target values for the wave.
                - feature_indices (list): The indices of the features that were extracted.
            Otherwise, a tuple containing two elements is returned:
                - X_wave (DataFrame): The input samples for the wave. Each row represents an observation,
                  and each column represents a feature.
                - y_wave (Series): The target values for the wave.

        Raises:
            ValueError: If the wave number is less than 0.

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
        """Fits the model to the given data.

        The following method iterates over each wave present in the dataset, retrieves the relevant data associated
        with each wave, and proceeds to train a replica of the classifier using said data. The trained classifier is
        subsequently appended to a pre-existing list.

        Once the processing of all waves is complete, the system proceeds to generate an ensemble classifier
        utilising the designated ensemble strategy. Subsequently, the ensemble classifier is trained on the provided
        training data for the final prediction of the class variable of new observations.

        Args:
            X (Union[List[List[float]], np.ndarray]):
                The input samples.
            y (Union[List[float], np.ndarray]):
                The target values.

        Returns:
            self: Returns self.

        Raises:
            ValueError: If the classifier, dataset, or feature groups are None, or if the ensemble strategy
              is neither 'voting' nor 'stacking'.

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
        """Predict class for X.

        The computation of the predicted class for an input sample involves determining the class that possesses the
        highest mean predicted probability. When the parameter 'voting' is set to 'hard', the predicted classes
        obtained from the underlying classifiers are utilised for the purpose of majority rule voting. When the value
        of the variable 'voting' is set to 'soft', the predicted probabilities of the underlying classifiers are
        summed together in order to determine the predicted class label.

        Args:
            X (Union[List[List[float]], np.ndarray)):
                The input samples.

        Returns:
            Union[List[float], np.ndarray]:
                The predicted classes.

        """
        if hasattr(self.clf_ensemble, "predict"):
            return self.clf_ensemble.predict(X)
        raise NotImplementedError(
            f"predict is not implemented for this classifier: {self.clf_ensemble} / type: {type(self.clf_ensemble)}"
        )

    @validate_predict_input
    def predict_proba(self, X: Union[List[List[float]], "np.ndarray"]) -> Union[List[List[float]], "np.ndarray"]:
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as the probabilities predicted by
        the underlying classifiers.

        Args:
            X (Union[List[List[float]], np.ndarray)):
                The input samples.

        Returns:
            Union[List[List[float]], np.ndarray]:
                The predicted class probabilities.

        """
        if hasattr(self.clf_ensemble, "predict_proba"):
            return self.clf_ensemble.predict_proba(X)
        raise NotImplementedError(
            "predict_proba is not implemented for this classifier: "
            f"{self.clf_ensemble} / type: {type(self.clf_ensemble)}"
        )

    @validate_predict_wave_input
    def predict_wave(self, wave: int, X: Union[List[List[float]], "np.ndarray"]) -> Union[List[float], "np.ndarray"]:
        """Predict class for X, using the classifier for the specified wave number.

        Args:
            wave (int):
                The wave number to extract.
            X (Union[List[List[float]], np.ndarray)):
                The input samples.

        Returns:
            Union[List[float], np.ndarray]:
                The predicted classes.

        """
        return self.estimators[wave][1].predict(X)
