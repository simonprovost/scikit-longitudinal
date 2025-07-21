# pylint: disable=too-many-arguments,invalid-name,signature-differs,no-member,R0801,protected-access
from functools import wraps
from typing import List, Optional, Union

import numpy as np
from overrides import override
from sklearn.utils.multiclass import unique_labels

from sklearn.ensemble import GradientBoostingClassifier
from scikit_longitudinal.templates import CustomClassifierMixinEstimator


def ensure_valid_state(method):
    """Decorator to ensure the classifier is in a valid state before method execution.

    This decorator performs checks on the classifier's attributes before executing certain methods
    that rely on the classifier being in a valid state. It ensures that the model is properly fitted
    and that all necessary configurations are set.

    The following checks are performed:
    - If the method name is 'predict' or 'predict_proba', it checks if the classifier has been fitted.
    - Checks if 'features_group' is set and contains more than one feature group.

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
        if method.__name__ in ["_predict", "_predict_proba"] and self._lexico_gradient_boosting is None:
            raise ValueError("The classifier must be fitted before calling predict or predict_proba.")

        if hasattr(self, "features_group") and (self.features_group is None or len(self.features_group) <= 1):
            raise ValueError("features_group must contain more than one feature group.")

        return method(self, *args, **kwargs)

    return wrapper


class LexicoGradientBoostingClassifier(CustomClassifierMixinEstimator):
    """
    Lexico Gradient Boosting Classifier for longitudinal data analysis.

    The Lexico Gradient Boosting Classifier is an advanced ensemble algorithm designed specifically for longitudinal
    datasets. It incorporates the fundamental principles of the Gradient Boosting framework while utilizing
    longitudinal-adapted base estimators to capture the temporal complexities and interdependencies intrinsic to
    longitudinal data. The base estimators are Lexico Decision Tree Regressors, which are specialized decision tree
    models capable of handling longitudinal data through a lexicographic optimization approach.

    !!! tip "Why Use LexicoGradientBoostingClassifier?"
        This classifier is ideal for longitudinal datasets where temporal recency is crucial. By leveraging lexicographic
        optimization within a boosting framework, it iteratively improves predictions while prioritizing recent
        measurements—perfect for applications like patient health monitoring or financial forecasting.

    !!! question "How Does Lexicographic Optimization Work?"
        The base estimators (Lexico Decision Tree Regressors) use a bi-objective split selection strategy:

        1. **Primary**: Minimize the loss (using "friedman_mse" criterion).
        2. **Secondary**: Favor features from more recent waves when loss reductions are similar (within `threshold_gain`).

        This ensures both statistical accuracy and temporal relevance are optimized, with boosting aggregating these
        decisions for enhanced predictive power.

    !!! note "Performance Boost with Cython"
        The underlying splitter (`node_lexicoRF_split`) is optimized in Cython for faster computation. See the
        [Cython implementation](https://github.com/simonprovost/scikit-lexicographical-trees/blob/21443b9dce51434b3198ccabac8bafc4698ce953/sklearn/tree/_splitter.pyx#L695)
        for details.

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

        To see more, we highly recommend visiting the `Temporal Dependency` page in the documentation.
        [Temporal Dependency Guide :fontawesome-solid-timeline:](https://scikit-longitudinal.readthedocs.io/latest/tutorials/temporal_dependency/){ .md-button }

    Args:
        threshold_gain (float, default=0.0015):
            Threshold for comparing loss reductions during split selection. Lower values enforce stricter recency
            preference; higher values allow more flexibility.
        features_group (List[List[int]], optional):
            Temporal matrix of feature indices for longitudinal attributes, ordered by recency. Required for longitudinal
            functionality.
        criterion (str, default="friedman_mse"):
            The split quality metric. Fixed to "friedman_mse"; do not modify.
        splitter (str, default="lexicoRF"):
            The split strategy. Fixed to "lexicoRF"; do not modify.
        max_depth (Optional[int], default=3):
            Maximum depth of each tree.
        min_samples_split (int, default=2):
            Minimum samples required to split an internal node.
        min_samples_leaf (int, default=1):
            Minimum samples required at a leaf node.
        min_weight_fraction_leaf (float, default=0.0):
            Minimum weighted fraction of total sample weight at a leaf.
        max_features (Optional[Union[int, str]], default=None):
            Number of features to consider for splits (e.g., "sqrt", "log2", int).
        random_state (Optional[int], default=None):
            Seed for random number generation.
        max_leaf_nodes (Optional[int], default=None):
            Maximum number of leaf nodes per tree.
        min_impurity_decrease (float, default=0.0):
            Minimum impurity decrease required for a split.
        ccp_alpha (float, default=0.0):
            Complexity parameter for pruning; non-negative.
        n_estimators (int, default=100):
            Number of boosting stages (trees) to perform.
        learning_rate (float, default=0.1):
            Learning rate shrinks the contribution of each tree. There is a trade-off between `learning_rate` and
            `n_estimators`.

    Attributes:
        _lexico_gradient_boosting (GradientBoostingClassifier):
            The underlying gradient boosting model.
        classes_ (ndarray):
            The class labels.

    Examples:
        !!! example "Basic Usage with Dummy Longitudinal Data"

            ```python
            from sklearn.metrics import accuracy_score
            from scikit_longitudinal.estimators.ensemble import LexicoGradientBoostingClassifier
            import numpy as np
            from scikit_longitudinal.data_preparation import LongitudinalDataset

            # Load dataset
            dataset = LongitudinalDataset('./stroke_longitudinal.csv')
            dataset.load_data()
            dataset.load_target(target_column="stroke_w2")
            dataset.setup_features_group("elsa")
            dataset.load_train_test_split(test_size=0.2, random_state=42)

            clf = LexicoGradientBoostingClassifier(features_group=dataset.feature_groups())
            clf.fit(dataset.X_train, dataset.y_train)
            y_pred = clf.predict(dataset.X_test)
            print(f"Accuracy: {accuracy_score(dataset.y_test, y_pred)}")
            ```

        !!! example "Tuning Learning Rate and Threshold Gain"

            ```python
            # ... Similar setup as above ...

            clf = LexicoGradientBoostingClassifier(
                features_group=[[0, 1], [2, 3]],
                threshold_gain=0.001, # Adjusted for hyperparameter tuning
                learning_rate=0.01, # Lower learning rate for more gradual learning
                n_estimators=200 # Increased number of estimators for better performance
            )
            clf.fit(X, y)
            y_pred = clf.predict(X)
            print(f"Accuracy: {accuracy_score(y, y_pred)}")

            # ... Similar evaluation as above ...
            ```

    Notes:
        - **References**:

          - Ribeiro, C. and Freitas, A., 2020. "A new random forest method for longitudinal data classification using a
            lexicographic bi-objective approach." *2020 IEEE Symposium Series on Computational Intelligence (SSCI)*,
            pp. 806-813.
          - Ribeiro, C. and Freitas, A.A., 2024. "A lexicographic optimisation approach to promote more recent features
            on longitudinal decision-tree-based classifiers." *Artificial Intelligence Review*, 57(4), p.84.
    """

    def __init__(
        self,
        threshold_gain: float = 0.0015,
        features_group: List[List[int]] = None,
        criterion: str = "friedman_mse",  # Do not change this value
        splitter: str = "lexicoRF",  # Do not change this value
        max_depth: Optional[int] = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[Union[int, str]] = None,
        random_state: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
    ):
        self.threshold_gain = threshold_gain
        self.features_group = features_group
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self._lexico_gradient_boosting = None
        self.classes_ = None

    @ensure_valid_state
    @override
    def _fit(self, X: np.ndarray, y: np.ndarray) -> "LexicoGradientBoostingClassifier":
        """Fit the Lexico Gradient Boosting Classifier model according to the given training data.

        Args:
            X (np.ndarray):
                The training input samples.
            y (np.ndarray):
                The target values (class labels).

        Returns:
            LexicoGradientBoostingClassifier: The fitted classifier.

        Raises:
            ValueError:
                If there are less than or equal to 1 feature group.

        !!! tip "Tuning Tip"
            Adjust `n_estimators` and `learning_rate` to balance model complexity and convergence speed. A lower
            `learning_rate` with more `n_estimators` can improve generalization but increases computation time.

        !!! note
            Ensure `features_group` accurately maps your data's temporal structure for optimal performance.
        """
        self._lexico_gradient_boosting = GradientBoostingClassifier(
            splitter=self.splitter,
            threshold_gain=self.threshold_gain,
            features_group=self.features_group,
        )

        if self.classes_ is None:
            self.classes_ = unique_labels(y)

        return self._lexico_gradient_boosting.fit(X, y)

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
            After fitting, use this method to generate predictions efficiently. It leverages the boosted ensemble for
            accurate classification.
        """
        return self._lexico_gradient_boosting.predict(X)

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
        return self._lexico_gradient_boosting.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return the feature importances.

        Returns:
            np.ndarray:
                The feature importances.

        !!! note
            Feature importances are calculated based on the impurity decrease across all trees in the ensemble.
        """
        return self._lexico_gradient_boosting.feature_importances_
