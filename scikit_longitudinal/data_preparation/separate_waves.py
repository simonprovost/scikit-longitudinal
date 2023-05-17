from typing import List, Union

import numpy as np
import ray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.exceptions import NotFittedError

from scikit_longitudinal.data_preparation import LongitudinalDataset


def validate_extract_wave_input(func):
    """Decorator to validate the input to the _extract_wave function.

    Args:
        func (function): Function to be decorated.

    Returns:
        function: Decorated function.

    Raises:
        ValueError: If the wave number is less than 0.

    """

    def wrapper(self, wave: int):
        if wave < 0:
            raise ValueError(f"Invalid wave number: {wave}. It should be more than 0")
        return func(self, wave)

    return wrapper


def validate_extract_wave_output(func):  # pragma: no cover
    """Decorator to validate the output of the _extract_wave function.

    Args:
        func (function): Function to be decorated.

    Returns:
        function: Decorated function.

    Raises:
        ValueError: If the number of features in the wave does not match the expected number of features.

    """

    def wrapper(self, wave: int):
        X_wave, y_wave = func(self, wave)
        expected_features = len([group[wave] for group in self.dataset.feature_groups() if wave < len(group)]) + len(
            self.dataset.non_longitudinal_features()
        )
        if X_wave.shape[1] != expected_features:
            raise ValueError(f"Invalid number of features in X_wave: {X_wave.shape[1]}. Expected {expected_features}.")
        return X_wave, y_wave

    return wrapper


def validate_fit_input(func):
    """Decorator to validate the input to the fit function.

    Args:
        func (function): Function to be decorated.

    Returns:
        function: Decorated function.

    Raises:
        ValueError: If the classifier, dataset, or feature groups are None.

    """

    def wrapper(self, X, y):
        if self.classifier is None or self.dataset is None or self.dataset.feature_groups() is None:
            raise ValueError("The classifier, dataset, and feature groups must not be None.")
        return func(self, X, y)

    return wrapper


def validate_fit_output(func):  # pragma: no cover
    """Decorator to validate the output of the fit function.

    Args:
        func (function): Function to be decorated.

    Returns:
        function: Decorated function.

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
        func (function): Function to be decorated.

    Returns:
        function: Decorated function.

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
        func (function): Function to be decorated.

    Returns:
        function: Decorated function.

    Raises:
        NotFittedError: If the SepWav instance is not fitted yet.

    """

    def wrapper(self, wave, X):
        if self.classifiers is None or len(self.classifiers) == 0:
            raise NotFittedError(
                "This SepWav instance is not fitted yet. Call 'fit' with appropriate arguments "
                "before using this estimator."
            )
        if not 0 <= wave < len(self.classifiers):
            raise IndexError(f"Invalid wave number: {wave}.")
        return func(self, wave, X)

    return wrapper


@ray.remote
def train_classifier(classifier, X_wave, y_wave, wave):  # pragma: no cover
    clf_wave = clone(classifier)
    clf_wave.fit(X_wave, y_wave)
    return f"wave_{wave}", clf_wave


# pylint: disable=too-many-instance-attributes, too-many-arguments
class SepWav(BaseEstimator, ClassifierMixin):
    """The SepWav (Separate Waves) class for data transformation in longitudinal data analysis.

    This technique involves supplying an algorithm defined by the user one wave at a time. Predicting a class
    variable, the algorithm is trained on a dataset containing only the current wave (beginning with the first). For
    each successive wave, the algorithm is trained using a newly-created dataset to which the new wave is added. This
    produces a number of classifiers. Then, these classifiers can be combined into an ensemble model or utilised in
    other ways based on user-defined methods. If Predict is invoked, the ensemble model will be used to predict the
    class variable. Nevertheless, if the user wishes to predict the class variable for a particular wave,
    they can invoke the predict_wave method. Or last way, merely by getting the list of classifiers into their script
    and playing with them. Ensemble strategies supported are voting and stacking.

    Attributes:
        dataset (LongitudinalDataset): The dataset to be used.
        classifier (BaseEstimator): The classifier to be used.
        classifiers (List): A list of classifiers for each wave.
        ensemble_strategy (str): The strategy for combining the classifiers ('voting' or 'stacking').
        clf_ensemble (BaseEstimator): The combined classifier.
        voting_weights (List[float]): The weights for the voting classifier.
        voting_strategy (str): The voting strategy for the voting classifier ('hard' or 'soft').
        n_jobs (int): The number of jobs to run in parallel.
        stacking_final_estimator (BaseEstimator): The final estimator to be used in the stacking classifier.
        stacking_cv (int): The cross-validation strategy for the stacking classifier.
        stacking_stack_method (str): The method to be used for the stacking classifier.
        stacking_passthrough (bool): Whether to pass the inputs to the final estimator for the stacking classifier.
        parallel (bool): Whether to run the fit waves in parallel.
        num_cpus (int): The number of cpus to use for parallel processing.

    Example:
        ```python
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        # Initialize the longitudinal data
        longitudinal_data = LongitudinalDataset("dementia_dataset.csv")
        longitudinal_data.load_data_target_train_test_split(
            target_column="class_dementia_w8",
            remove_target_waves=True,
        )
        longitudinal_data.setup_features_group(input_data="elsa")

        # Initialize the classifier
        classifier = RandomForestClassifier()

        # Initialize the SepWav instance
        sepwav = SepWav(dataset=longitudinal_data, classifier=classifier)

        # Fit and predict
        sepwav.fit(longitudinal_data.X_train, longitudinal_data.y_train)
        y_pred = sepwav.predict(longitudinal_data.X_test)
        print(classification_report(longitudinal_data.y_test, y_pred))
        ```

    """

    def __init__(
        self,
        dataset: "LongitudinalDataset",
        classifier: "BaseEstimator",
        ensemble_strategy: str = "voting",
        voting_weights: List[float] = None,
        voting_strategy: str = "hard",
        n_jobs: int = None,
        stacking_final_estimator: "BaseEstimator" = None,
        stacking_cv: int = None,
        stacking_stack_method: str = "auto",
        stacking_passthrough: bool = False,
        parallel: bool = False,
        num_cpus: int = -1,
    ):
        self.dataset = dataset
        self.classifier = classifier
        self.classifiers = []
        self.ensemble_strategy = ensemble_strategy
        self.clf_ensemble = None
        self.voting_weights = voting_weights
        self.voting_strategy = voting_strategy
        self.n_jobs = n_jobs
        self.stacking_final_estimator = stacking_final_estimator
        self.stacking_cv = stacking_cv
        self.stacking_stack_method = stacking_stack_method
        self.stacking_passthrough = stacking_passthrough
        self.parallel = parallel
        self.num_cpus = num_cpus

        if self.parallel and ray.is_initialized() is False:  # pragma: no cover
            if self.num_cpus != -1:
                ray.init(num_cpus=self.num_cpus)
            else:
                ray.init()

    @validate_extract_wave_input
    @validate_extract_wave_output
    def _extract_wave(self, wave: int) -> tuple:
        """Extracts a specific wave from the dataset for training.

        This method uses the wave number to select specific features from the dataset for training.
        It identifies the features that correspond to the wave, extends it with non-longitudinal features,
        and slices the dataset to create the training data for the wave.

        Args:
            wave (int): The wave number to extract. This should be a non-negative integer.

        Returns:
            tuple: A tuple containing two elements:
                - X_wave (DataFrame): The input samples for the wave. Each row represents an observation,
                  and each column represents a feature.
                - y_wave (Series): The target values for the wave.

        Raises:
            ValueError: If the wave number is less than 0.

        """
        feature_indices = [group[wave] for group in self.dataset.feature_groups() if wave < len(group)]
        feature_indices.extend(self.dataset.non_longitudinal_features())

        X_wave = self.dataset.X_train.iloc[:, feature_indices]
        y_wave = self.dataset.y_train

        return X_wave, y_wave

    # pylint: disable=unused-argument
    @validate_fit_input
    @validate_fit_output
    def fit(self, X: Union[List[List[float]], "np.ndarray"], y: Union[List[float], "np.ndarray"]):
        """Fits the model to the given data.

        This method goes through each wave in the dataset, extracts the corresponding data, and trains a clone
        of the classifier on it. It then adds the trained classifier to a list.

        After all waves have been processed, it creates an ensemble classifier using the specified ensemble strategy,
        and fits it on the training data.

        Args:
            X (Union[List[List[float]], np.ndarray]): The input samples. This can be a list of lists or a numpy array,
              where each inner list or sub-array represents an observation, and its elements represent the features.
            y (Union[List[float], np.ndarray]): The target values. This can be a list or a numpy array.

        Returns:
            self: Returns self.

        Raises:
            ValueError: If the classifier, dataset, or feature groups are None, or if the ensemble strategy
              is neither 'voting' nor 'stacking'.

        """
        if self.parallel:
            futures = [
                train_classifier.remote(self.classifier, X_train, y_train, wave)
                for wave, (X_train, y_train) in enumerate(
                    self._extract_wave(wave=i)
                    for i in range(max(len(group) for group in self.dataset.feature_groups()))
                )
            ]

            self.classifiers = ray.get(futures)
        else:
            for i in range(max(len(group) for group in self.dataset.feature_groups())):
                X_wave, y_wave = self._extract_wave(wave=i)
                clf_wave = clone(self.classifier)
                clf_wave.fit(X_wave, y_wave)
                self.classifiers.append((f"wave_{i}", clf_wave))

        if self.ensemble_strategy == "voting":
            self.clf_ensemble = VotingClassifier(
                self.classifiers, voting=self.voting_strategy, weights=self.voting_weights, n_jobs=self.n_jobs
            )
        elif self.ensemble_strategy == "stacking":
            self.clf_ensemble = StackingClassifier(self.classifiers, final_estimator=self.classifier)
        else:
            raise ValueError(f"Invalid ensemble strategy: {self.ensemble_strategy}")

        self.clf_ensemble.fit(self.dataset.X_train, self.dataset.y_train)

        return self

    @validate_predict_input
    def predict(self, X: Union[List[List[float]], "np.ndarray"]) -> Union[List[float], "np.ndarray"]:
        """Predict class for X.

        The predicted class of an input sample is computed as the class with the highest mean predicted probability.
        If ‘voting=’hard’, the predicted classes of the underlying classifiers are used for majority rule voting. If
        ‘voting=’soft’, sums predicted probabilities of the underlying classifiers are used to predict the class label.

        Args:
            X (Union[List[List[float]], np.ndarray)): The input samples.

        Returns:
            Union[List[float], np.ndarray]: The predicted classes.

        """
        return self.clf_ensemble.predict(X)

    @validate_predict_wave_input
    def predict_wave(self, wave: int, X: Union[List[List[float]], "np.ndarray"]) -> Union[List[float], "np.ndarray"]:
        """Predict class for X, using the number of the wave.

        Args:
            wave (int): The wave number to extract.
            X (Union[List[List[float]], np.ndarray)): The input samples.

        Returns:
            Union[List[float], np.ndarray]: The predicted classes.

        """
        return self.classifiers[wave][1].predict(X)