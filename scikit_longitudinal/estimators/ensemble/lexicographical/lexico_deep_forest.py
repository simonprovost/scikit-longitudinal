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

from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_random_forest import (
    LexicoRandomForestClassifier,
)
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
        if (
            method.__name__ in ["_predict", "_predict_proba"]
            and self._deep_forest is None
        ):
            raise ValueError(
                "The classifier must be fitted before calling predict or predict_proba."
            )

        if hasattr(self, "features_group") and (
            self.features_group is None or len(self.features_group) <= 1
        ):
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
    """
    Lexico Deep Forest Classifier for longitudinal data analysis.

    This classifier extends the Deep Forest framework for longitudinal data by stacking layers of
    longitudinal-adapted base estimators (typically `LexicoRandomForestClassifier`) so each layer's predictions
    become additional features for the next. Every base tree applies a lexicographic split-selection rule: the
    primary objective maximises the information-gain ratio (entropy criterion), and the secondary objective
    favours features from more recent waves whenever competing gain ratios are within `threshold_gain`. For more
    information on Deep Forest, see [DF21](https://deep-forest.readthedocs.io/en/stable/).

    Args:
        features_group (List[List[int]], optional):
            Temporal matrix of feature indices for longitudinal attributes, ordered by recency. Required for longitudinal
            functionality.
        longitudinal_base_estimators (Optional[List[LongitudinalEstimatorConfig]], optional):
            List of configurations for longitudinal base estimators. Each config specifies the classifier type, count,
            and optional hyperparameters. Available types: `LEXICO_RF`, `COMPLETE_RANDOM_LEXICO_RF`.
        non_longitudinal_features (List[Union[int, str]], optional):
            Indices of non-longitudinal features. Defaults to None.
        diversity_estimators (bool, default=True):
            Whether to include diversity estimators (weak learners) in the ensemble. If True, two completely random
            `LexicoRandomForestClassifier` instances are added.
        class_weight (Optional[Union[dict, List[dict], str]]):
            Class weights passed to each longitudinal base estimator unless explicitly provided in the estimator's
            hyperparameters.
        random_state (int, optional):
            Seed for random number generation. Defaults to None.
        single_classifier_type (Optional[Union[LongitudinalClassifierType, str]], optional):
            Type of a single classifier to use if `longitudinal_base_estimators` is not provided.
        single_count (Optional[int], optional):
            Number of instances of the single classifier type.
        max_layers (int, default=5):
            Maximum number of cascade layers in the deep forest.

    Attributes:
        _deep_forest (CascadeForestClassifier):
            The underlying deep forest model.
        classes_ (ndarray):
            The class labels.

    Examples:
        !!! example "Basic Usage"

            ```python
            from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_deep_forest import LexicoDeepForestClassifier, \
                LongitudinalEstimatorConfig, LongitudinalClassifierType
            import numpy as np
            from sklearn.metrics import accuracy_score
            from scikit_longitudinal.data_preparation import LongitudinalDataset

            # Load dataset
            dataset = LongitudinalDataset('./stroke_longitudinal.csv')
            dataset.load_data()
            dataset.load_target(target_column="stroke_w2")
            dataset.setup_features_group("elsa")
            dataset.load_train_test_split(test_size=0.2, random_state=42)


            # Configure base estimators
            lexico_rf_config = LongitudinalEstimatorConfig(
                classifier_type=LongitudinalClassifierType.LEXICO_RF,
                count=3,
            )

            clf = LexicoDeepForestClassifier(
                features_group=dataset.feature_groups(),
                longitudinal_base_estimators=[lexico_rf_config],
            )

            clf.fit(dataset.X_train, dataset.y_train)
            y_pred = clf.predict(dataset.X_train)
            print(f"Predictions: {y_pred}")
            ```

        !!! example "Advanced: multiple estimator types"

            ```python
            # ... Similar setup as above ...

            complete_random_lexico_rf = LongitudinalEstimatorConfig(
                classifier_type=LongitudinalClassifierType.COMPLETE_RANDOM_LEXICO_RF,
                count=2,
            )
            clf = LexicoDeepForestClassifier(
                features_group=features_group,
                longitudinal_base_estimators=[lexico_rf_config, complete_random_lexico_rf],
            )
            clf.fit(X, y)

            # ... Similar prediction and evaluation as above ...
            ```

        !!! example "Advanced: disabling diversity estimators"

            ```python
            # ... Similar setup as above ...

            clf = LexicoDeepForestClassifier(
                features_group=features_group,
                longitudinal_base_estimators=[lexico_rf_config],
                diversity_estimators=False, # Disable diversity estimators
            )
            clf.fit(X, y)

            # ... Similar prediction and evaluation as above ...
            ```
    """

    # pylint: disable=too-many-arguments,invalid-name,signature-differs,no-member
    def __init__(
        self,
        features_group: List[List[int]] = None,
        longitudinal_base_estimators: Optional[
            List[LongitudinalEstimatorConfig]
        ] = None,
        non_longitudinal_features: List[Union[int, str]] = None,
        diversity_estimators: bool = True,
        class_weight: Optional[Union[dict, List[dict], str]] = None,
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
        self.class_weight = class_weight
        self.random_state = random_state
        self._deep_forest = None
        self.classes_ = None
        self.max_layers = max_layers

    @property
    def base_longitudinal_estimators(self) -> List[ClassifierMixin]:
        estimators: List[ClassifierMixin] = []
        for estimator_info in self.longitudinal_base_estimators:
            base_hyperparameters = estimator_info.hyperparameters or {}
            for _ in range(estimator_info.count):
                estimators.append(
                    self._create_longitudinal_estimator(
                        estimator_info.classifier_type, **dict(base_hyperparameters)
                    )
                )

        if self.diversity_estimators:
            for _ in range(2):
                estimators.append(
                    self._create_longitudinal_estimator(
                        LongitudinalClassifierType.COMPLETE_RANDOM_LEXICO_RF
                    )
                )
        return estimators

    def _create_longitudinal_estimator(
        self,
        classifier_type: Union[str, LongitudinalClassifierType],
        **hyperparameters: Any,
    ) -> ClassifierMixin:
        resolved_hyperparameters = dict(hyperparameters)
        if (
            "class_weight" not in resolved_hyperparameters
            and self.class_weight is not None
        ):
            resolved_hyperparameters["class_weight"] = self.class_weight

        if classifier_type in {
            LongitudinalClassifierType.LEXICO_RF,
            LongitudinalClassifierType.LEXICO_RF.value,
        }:
            return LexicoRandomForestClassifier(
                features_group=self.features_group, **resolved_hyperparameters
            )
        if classifier_type in {
            LongitudinalClassifierType.COMPLETE_RANDOM_LEXICO_RF,
            LongitudinalClassifierType.COMPLETE_RANDOM_LEXICO_RF.value,
        }:
            resolved_hyperparameters.setdefault("max_features", 1)
            return LexicoRandomForestClassifier(
                features_group=self.features_group, **resolved_hyperparameters
            )
        raise ValueError(f"Unsupported classifier type: {classifier_type.value}")

    @ensure_valid_state
    @override
    def _fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight=None
    ) -> "LexicoDeepForestClassifier":
        """Fit the Lexico Deep Forest Classifier model according to the given training data.

        Args:
            X (np.ndarray):
                The training input samples.
            y (np.ndarray):
                The target values (class labels).

        Returns:
            LexicoDeepForestClassifier: The fitted classifier.

        Raises:
            ValueError:
                If there are less than or equal to 1 feature group.

        !!! tip "Configuration Tip"
            Experiment with different combinations of `longitudinal_base_estimators` and `diversity_estimators` to
            find the optimal balance between accuracy and diversity for your dataset.
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
        self._deep_forest.fit(X, y, sample_weight=sample_weight)
        self.classes_ = getattr(self._deep_forest, "classes_", unique_labels(y))
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

        !!! tip "Quick Predictions"
            After fitting, use this method to generate predictions efficiently. It leverages the deep forest ensemble for
            accurate classification.
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
