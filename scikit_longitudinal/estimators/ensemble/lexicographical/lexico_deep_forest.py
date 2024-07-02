# pylint: disable=R0902,R0903,R0914,R0801,too-many-arguments,invalid-name,signature-differs,no-member, W0212

from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Union

import numpy as np
from deepforest import CascadeForestClassifier
from overrides import override
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import unique_labels

from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_random_forest import LexicoRandomForestClassifier
from scikit_longitudinal.templates import CustomClassifierMixinEstimator


def ensure_valid_state(method):
    """Decorator to ensure the classifier is in a valid state before method execution.

    This decorator performs checks on the classifier's attributes before executing certain methods
    that rely on the classifier being in a valid state. It ensures that the model is properly fitted
    and that all necessary configurations are set.

    The following checks are performed:
    - If the method name is 'predict' or 'predict_proba', it checks if the classifier has been fitted.
    - Checks if 'features_group' is set and contains more than one feature group.
    - Ensures that 'longitudinal_base_estimators' has been provided.
    - Verifies that 'diversity_estimators' has been explicitly set to True or False.

    If any of these checks fail, a ValueError is raised with an appropriate error message.

    Args:
        method (function):
            The method to be wrapped by the decorator.

    Returns:
        function:
            The wrapped method with pre-execution validation.

    Raises:
        ValueError: If any of the checks fail, indicating that the classifier is not in a valid state.

    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if method.__name__ in ["_predict", "_predict_proba"] and self._deep_forest is None:
            raise ValueError("The classifier must be fitted before calling predict or predict_proba.")

        if hasattr(self, "features_group") and (self.features_group is None or len(self.features_group) <= 1):
            raise ValueError("features_group must contain more than one feature group.")

        if hasattr(self, "diversity_estimators") and self.diversity_estimators is None:
            raise ValueError("diversity_estimators must be provided. True or False.")

        return method(self, *args, **kwargs)

    return wrapper


class LongitudinalClassifierType(Enum):
    """Enumeration of classifier types that are adapted for longitudinal data analysis.

    This enumeration provides identifiers for longitudinal-adapted classifiers that can be used within the
    LexicoDeepForestClassifier ensemble.

    Attributes:
        LEXICO_RF: Identifier for a Lexico Random Forest Classifier.
        COMPLETE_RANDOM_LEXICO_RF: Identifier for a Lexico Random Forest Classifier with complete randomness.

    """

    LEXICO_RF = "LexicoRandomForestClassifier"
    COMPLETE_RANDOM_LEXICO_RF = "LexicoCompleteRFClassifier"


@dataclass
class LongitudinalEstimatorConfig:
    """Configuration for a longitudinal base estimator within the LexicoDeepForestClassifier ensemble.

    This configuration class is used to specify the type of longitudinal classifier, the number of times it should be
    instantiated within the ensemble, and any hyperparameters for the individual classifiers.

    Args:
        classifier_type (LongitudinalClassifierType):
            The type of longitudinal classifier to be used.
        count (int):
            The number of times the classifier should be replicated in the ensemble. Defaults to 2.
        hyperparameters (Optional[Dict[str, Any]]):
            A dictionary of hyperparameters for the classifier. Defaults to None.

    """

    classifier_type: LongitudinalClassifierType
    count: int = 2
    hyperparameters: Optional[Dict[str, Any]] = None


class LexicoDeepForestClassifier(CustomClassifierMixinEstimator):
    """Deep Forest Classifier adapted for longitudinal data analysis.

    ⚠️ Scikit-Longitudinal's docstrings will be updated to reflect the most recent documentation available on Github.
    If something is inconsistent, consult the documentation first, then file an issue. ⚠️

    Deep Forests Longitudinal Classifier is an advanced ensemble algorithm designed specifically for longitudinal
    datasets, which incorporates the fundamental principles of the Deep Forests framework itself. This classifier
    distinguishes itself through the implementation of longitudinal-adapted base estimators, which are intended to
    capture the temporal complexities and interdependencies that are intrinsic to longitudinal data.

    The classifier ensemble is composed of two types of estimators:

    1. Accurate Learners (longitudinal base estimators): These are the primary estimators that form the backbone
       of the ensemble. They are adapted from conventional machine learning algorithms to better handle the
       temporal aspect of longitudinal data. Currently, the following base estimators are supported:
       - LexicoRandomForestClassifier
       - LexicoCompleteRFClassifier
    2. Weak Learners (diversity estimators): In addition to the accurate learners, the ensemble includes
       diversity estimators to enhance the overall diversity of the model. These estimators, which are typically
       less complex and may have a higher bias, contribute unique perspectives to the decision-making process,
       thereby improving the ensemble's robustness and generalization capabilities. When enabled, the diversity
       estimators include two completely random LexicoRFClassifiers. Readers are referred to the paper for more
       information on the diversity estimators.

    The combination of these accurate and weak learners aims to exploit the strengths of each estimator type,
    leading to a more effective and reliable classification performance on longitudinal datasets.

    Args:
        features_group (List[List[int]]):
            A list of lists, where each inner list contains the indices of features that
            correspond to a specific longitudinal attribute. This parameter will be forwarded to the base
            longitudinal-based(-adapted) algorithms, if required.
        longitudinal_base_estimators (List[LongitudinalEstimatorConfig]):
            A list of `LongitudinalEstimatorConfig` objects that define the configuration for each base estimator
            within the ensemble. Each configuration specifies the type of longitudinal classifier, the number of
            times it should be instantiated within the ensemble, and an optional dictionary of hyperparameters
            for finer control over the individual classifiers' behavior. Available longitudinal classifiers are:
            - LEXICO_RF
            - COMPLETE_RANDOM_LEXICO_RF
        non_longitudinal_features (List[Union[int, str]], optional):
            A list of indices of features that are not longitudinal attributes. Defaults to None. This parameter will be
            forwarded to the base longitudinal-based(-adapted) algorithms if required.
        diversity_estimators (bool, optional):
            A flag indicating whether the ensemble should include diversity estimators, defaulting to True. When
            enabled, diversity estimators, which function as weak learners, are added to the ensemble to enhance
            its diversity and, by extension, its predictive performance. Disabling this option results in an ensemble
            comprising solely of the specified base longitudinal-adapted algorithms. The diversity is achieved by
            integrating two additional completely random LexicoRandomForestClassifier instances into the ensemble.
        random_state (int, optional):
            The seed used by the random number generator. Defaults to None.

    Examples:
        # Example with specific count and hyperparameters for a single type of longitudinal estimator
        >>> from deep_forest import DeepForestsClassifier
        >>> X = <your_training_data>  # Replace with your actual data
        >>> y = <your_training_target_data>  # Replace with your actual target
        >>> features_group = <your_features_group>  # Construct this based on your LongitudinalDataset
        >>> non_longitudinal_features = <your_non_longitudinal_features>  # Similarly here
        >>> lexico_rf_config = LongitudinalEstimatorConfig(
        ...     classifier_type=LongitudinalClassifierType.LEXICO_RF,
        ...     count=3,
        ...     hyperparameters={'max_depth': 5, 'n_estimators': 10}
        ... )
        >>> clf = LexicoDeepForestClassifier(
        ...     features_group=features_group,
        ...     non_longitudinal_features=non_longitudinal_features,
        ...     longitudinal_base_estimators=[lexico_rf_config],
        ...     random_state=42
        ... )
        >>> clf.fit(X, y)
        >>> clf.predict(X)

        # Example with multiple types of longitudinal estimators
        >>> complete_random_lexico_rf = LongitudinalEstimatorConfig(
        ...     classifier_type=LongitudinalClassifierType.COMPLETE_RANDOM_LEXICO_RF,
        ...     count=2,
        ...     hyperparameters={'max_depth': 3, 'n_estimators': 5}
        ... )
        >>> clf = LexicoDeepForestClassifier(
        ...     features_group=features_group,
        ...     non_longitudinal_features=non_longitudinal_features,
        ...     longitudinal_base_estimators=[lexico_rf_config, complete_random_lexico_rf],
        ...     random_state=42
        ... )
        >>> clf.fit(X, y)
        >>> clf.predict(X)

        # Example without specifying count and hyperparameters, using default values
        >>> clf = LexicoDeepForestClassifier(
        ...     features_group=features_group,
        ...     non_longitudinal_features=non_longitudinal_features,
        ...     longitudinal_base_estimators=[
        ...         LongitudinalEstimatorConfig(classifier_type=LongitudinalClassifierType.LEXICO_RF),
        ...         LongitudinalEstimatorConfig(classifier_type=LongitudinalClassifierType.COMPLETE_RANDOM_LEXICO_RF)
        ...     ],
        ...     random_state=42
        ... )
        >>> clf.fit(X, y)
        >>> clf.predict(X)

        # Example with diversity estimators disabled
        >>> clf = LexicoDeepForestClassifier(
        ...     features_group=features_group,
        ...     non_longitudinal_features=non_longitudinal_features,
        ...     longitudinal_base_estimators=[lexico_rf_config],
        ...     diversity_estimators=False,
        ...     random_state=42
        ... )
        >>> clf.fit(X, y)
        >>> clf.predict(X)

    Notes:
        For more information, see the following paper of the Deep Forest algorithm:

        Zhou, Z.H. and Feng, J., 2019. Deep forest. National science review, 6(1), pp.74-86.


        Here is the initial Python implementation of the Deep Forest algorithm:
        https://github.com/LAMDA-NJU/Deep-Forest

     See Also:
        CustomClassifierMixinEstimator: Base class for all Classifier Mixin estimators in scikit-learn that we
            customized so that the original scikit-learn "check_x_y" is performed all the time.

    """

    # pylint: disable=too-many-arguments,invalid-name,signature-differs,no-member
    def __init__(
        self,
        features_group: List[List[int]] = None,
        longitudinal_base_estimators: Optional[List[LongitudinalEstimatorConfig]] = None,
        non_longitudinal_features: List[Union[int, str]] = None,
        diversity_estimators: bool = True,
        random_state: int = None,
        single_classifier_type: Optional[Union[LongitudinalClassifierType, str]] = None,
        single_count: Optional[int] = None,
        max_layers: int = 5,
    ):
        self.features_group = features_group
        self.non_longitudinal_features = non_longitudinal_features
        self.single_classifier_type = single_classifier_type
        self.single_count = single_count
        self.longitudinal_base_estimators = longitudinal_base_estimators
        self.diversity_estimators = diversity_estimators
        self.random_state = random_state
        self._deep_forest = None
        self.classes_ = None
        self.max_layers = max_layers

    @property
    def base_longitudinal_estimators(self) -> List[ClassifierMixin]:
        estimators = [
            self._create_longitudinal_estimator(estimator_info.classifier_type, **estimator_info.hyperparameters or {})
            for estimator_info in self.longitudinal_base_estimators
            for _ in range(estimator_info.count)
        ]

        if self.diversity_estimators:
            estimators.extend(
                self._create_longitudinal_estimator(
                    LongitudinalClassifierType.COMPLETE_RANDOM_LEXICO_RF,
                )
                for _ in range(2)
            )
        return estimators

    def _create_longitudinal_estimator(
        self, classifier_type: Union[str, LongitudinalClassifierType], **hyperparameters: Any
    ) -> ClassifierMixin:
        if classifier_type in {LongitudinalClassifierType.LEXICO_RF, LongitudinalClassifierType.LEXICO_RF.value}:
            return LexicoRandomForestClassifier(features_group=self.features_group, **hyperparameters)
        if classifier_type in {
            LongitudinalClassifierType.COMPLETE_RANDOM_LEXICO_RF,
            LongitudinalClassifierType.COMPLETE_RANDOM_LEXICO_RF.value,
        }:
            return LexicoRandomForestClassifier(features_group=self.features_group, max_features=1, **hyperparameters)
        raise ValueError(f"Unsupported classifier type: {classifier_type.value}")

    @ensure_valid_state
    @override
    def _fit(self, X: np.ndarray, y: np.ndarray) -> "LexicoDeepForestClassifier":
        """Fit the Deep Forest Longitudinal Classifier model according to the given training data.

        Args:
            X (np.ndarray):
                The training input samples.
            y (np.ndarray):
                The target values (class labels).

        Returns:
            NestedTreesClassifier: The fitted classifier.

        Raises:
            ValueError:
                If there are less than or equal to 1 feature group.

        """
        if self.single_classifier_type is not None and self.single_count is not None:
            self.longitudinal_base_estimators = [
                LongitudinalEstimatorConfig(
                    classifier_type=self.single_classifier_type,
                    count=self.single_count,
                )
            ]
        elif self.longitudinal_base_estimators is None:
            raise ValueError("longitudinal_base_estimators must be provided.")
        if self.features_group is None or len(self.features_group) <= 1:
            raise ValueError("features_group must contain more than one feature group.")
        self._deep_forest = CascadeForestClassifier(
            random_state=self.random_state,
            max_layers=self.max_layers,
        )
        self._deep_forest.set_estimator(self.base_longitudinal_estimators, n_splits=2)
        if self.classes_ is None:
            self.classes_ = unique_labels(y)
        self._deep_forest.fit(X, y)
        return self

    @ensure_valid_state
    @override
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        Args:
            X (np.ndarray):
                The input samples.

        Returns:
            np.ndarray:
                The predicted class labels for each input sample.

        """
        return self._deep_forest.predict(X)

    @ensure_valid_state
    @override
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X.

        Args:
            X (np.ndarray):
                The input samples.

        Returns:
            np.ndarray:
                The predicted class probabilities for each input sample.

        """
        return self._deep_forest.predict_proba(X)
