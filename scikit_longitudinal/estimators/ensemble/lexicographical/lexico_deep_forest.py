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
    """
    Lexico Deep Forest Classifier for longitudinal data analysis.

    The Lexico Deep Forest Classifier is an advanced ensemble algorithm designed specifically for longitudinal data
    analysis. It extends the fundamental principles of the Deep Forest framework by incorporating longitudinal-adapted
    base estimators to capture the temporal complexities and interdependencies inherent in longitudinal data. The
    classifier combines accurate learners (longitudinal base estimators) and weak learners (diversity estimators) to
    improve robustness and generalization.

    !!! tip "Why Use LexicoDeepForestClassifier?"
        This classifier is ideal for longitudinal datasets where temporal structure is crucial. By leveraging a deep
        forest architecture with longitudinal-adapted estimators, it captures complex patterns and temporal dependencies
        effectively—perfect for applications like medical studies or time-series classification.

    !!! question "How Does It Work?"
        The classifier builds a cascade of forests, where each layer uses the predictions from the previous layer as
        additional features. The base estimators are longitudinal-adapted classifiers like `LexicoRandomForestClassifier`,
        which use lexicographic optimization to prioritize recent data points. Diversity estimators (weak learners) are
        optionally included to enhance the ensemble's diversity and predictive performance.

    !!! note "Performance Boost with Cython"
        The underlying decision trees use a Cython-optimized splitter (`node_lexicoRF_split`) for faster computation.
        See the [Cython implementation](https://github.com/simonprovost/scikit-lexicographical-trees/blob/21443b9dce51434b3198ccabac8bafc4698ce953/sklearn/tree/_splitter.pyx#L695)
        for details.

    !!! question "Feature Groups and Non-Longitudinal Features"
        Two key attributes define the temporal structure:

        - **features_group**: A list of lists, each sublist containing indices of a longitudinal attribute's waves,
          ordered from oldest to most recent (e.g., `[[0,1], [2,3]]` for two attributes with two waves each).
        - **non_longitudinal_features**: Indices of static features (not used in lexicographic optimization but included
          in standard splits).

        Accurate configuration is essential for leveraging temporal patterns. See the
        [Temporal Dependency Guide](https://scikit-longitudinal.readthedocs.io/latest/tutorials/temporal_dependency/) for more.

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
        !!! example "Basic Usage with LexicoRandomForestClassifier"

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

        !!! example "Using Multiple Estimator Types"

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

        !!! example "Disabling Diversity Estimators"

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

    Notes:
        - **References**:

          - Zhou, Z.H. and Feng, J., 2019. "Deep forest." *National Science Review*, 6(1), pp.74-86.
          - [Deep Forest GitHub](https://github.com/LAMDA-NJU/Deep-Forest)
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

        !!! note
            Ensure `features_group` accurately maps your data's temporal structure for optimal performance.
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

        !!! question "When to Use Probabilities?"
            Use `predict_proba` instead of `predict` when you need to assess confidence levels or apply custom
            decision thresholds rather than relying on the default class assignment.
        """
        return self._deep_forest.predict_proba(X)
