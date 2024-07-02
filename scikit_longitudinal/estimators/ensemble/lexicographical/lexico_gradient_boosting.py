# pylint: disable=too-many-arguments,invalid-name,signature-differs,no-member,R0801,protected-access
from functools import wraps
from typing import List, Optional, Union

import numpy as np
from overrides import override
from sklearn.utils.multiclass import unique_labels
from starboost import BoostingClassifier

from scikit_longitudinal.estimators.trees.lexicographical.lexico_decision_tree_regressor import (
    LexicoDecisionTreeRegressor,
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
    """Gradient Boosting Classifier adapted for longitudinal data analysis.

    ⚠️ Scikit-Longitudinal's docstrings will be updated to reflect the most recent documentation available on Github.
    If something is inconsistent, consult the documentation first, then file an issue. ⚠️

    Gradient Boosting Longitudinal Classifier is an advanced ensemble algorithm designed specifically for longitudinal
    datasets, which incorporates the fundamental principles of the Gradient Boosting framework itself. This classifier
    distinguishes itself through the implementation of longitudinal-adapted base estimators, which are intended to
    capture the temporal complexities and interdependencies that are intrinsic to longitudinal data.

    That is, the base estimators of the Gradient Boosting Longitudinal Classifier are Lexico Decision Trees Regressors,
    which are specialised decision tree models that are capable of handling longitudinal data.

    Args:
        threshold_gain : float
            The threshold value for comparing gain ratios of features during the decision tree construction.
        features_group : List[List[int]]
            A list of lists, where each inner list contains the indices of features that
            correspond to a specific longitudinal attribute.
        criterion : str, optional (default="friedman_mse")
            The function to measure the quality of a split. Do not change this value.
        splitter : str, optional (default="lexicoRF")
            The strategy used to choose the split at each node. Do not change this value.
        max_depth : int, optional (default=None)
            The maximum depth of the tree.
        min_samples_split : int, optional (default=2)
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int, optional (default=1)
            The minimum number of samples required to be at a leaf node.
        min_weight_fraction_leaf : float, optional (default=0.)
            The minimum weighted fraction of the sum total of weights required to be at a leaf node.
        max_features : int, optional (default=None)
            The number of features to consider when looking for the best split.
        random_state : int, optional (default=None)
            The seed used by the random number generator.
        max_leaf_nodes : int, optional (default=None)
            The maximum number of leaf nodes in the tree.
        min_impurity_decrease : float, optional (default=0.)
            The minimum impurity decrease required for a node to be split.
        ccp_alpha : float, optional (default=0.0)
            Complexity parameter used for Minimal Cost-Complexity Pruning.

    Examples:
        >>> from scikit_longitudinal.estimators.ensemble.lexicographical import LexicoGradientBoostingClassifier
        >>> X = <your_training_data>  # Replace with your actual data
        >>> y = <your_training_target_data>  # Replace with your actual target
        >>> features_group = <your_features_group>  # Construct this based on your LongitudinalDataset
        >>> clf = LexicoGradientBoostingClassifier(
        ...     features_group=features_group,
        ...     threshold_gain=0.0015,
        ...     max_depth=3,
        ...     random_state=42
        ... )
        >>> clf.fit(X, y)
        >>> clf.predict(X)

    Notes:
    For more information, please refer to the following paper:

    Ribeiro, C. and Freitas, A., 2020, December. A new random forest method for longitudinal data
    regression using a lexicographic bi-objective approach. In 2020 IEEE Symposium Series on


     See Also:
        CustomClassifierMixinEstimator: Base class for all Classifier Mixin estimators in scikit-learn that we
            customized so that the original scikit-learn "check_x_y" is performed all the time.

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
        tree_flavor: bool = False,
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
        self.tree_flavor = tree_flavor
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self._lexico_gradient_boosting = None
        self._base_estimator = None
        self.classes_ = None

    @ensure_valid_state
    @override
    def _fit(self, X: np.ndarray, y: np.ndarray) -> "LexicoGradientBoostingClassifier":
        """Fit the Lexico Gradient Boost Longitudinal Classifier model according to the given training data.

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
        _base_estimator = LexicoDecisionTreeRegressor(
            features_group=self.features_group,
            threshold_gain=self.threshold_gain,
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
        )

        self._base_estimator = _base_estimator
        self._lexico_gradient_boosting = BoostingClassifier(
            base_estimator=_base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            tree_flavor=self.tree_flavor,
            random_state=self.random_state,
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

        """
        return self._lexico_gradient_boosting.predict_proba(X)
