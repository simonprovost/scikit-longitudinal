from itertools import combinations
from typing import List

import numpy as np

from scikit_longitudinal.preprocessors.feature_selection.correlation_feature_selection.operations import (
    _FeatureSelectorCache,
    merit_calculation,
)


def _exhaustive_search(X: np.ndarray, y: np.ndarray) -> List[int]:
    """Performs an exhaustive search to find the best feature set based on merit score.

    This function conducts an exhaustive search over all possible combinations of feature
    indices to identify the set that maximizes the merit score. The merit score is computed
    based on the correlation between features and between each feature and the class variable.
    This exhaustive approach ensures that the best possible set of features is selected,
    albeit with potentially high computational cost for datasets with a large number of features.

    Note on Thus Far Tested Experiments: With a dataset of over 7000 instances and more than 50 features to consider,
    this is far from "doable". Below 50 features for the same number of instances may take some time, but it is
    "doable." However, keep in mind that its exhaustive nature requires a significant amount of time overall, hence
    consider lower number of features for faster computation.

    Args:
        X (np.ndarray):
            The feature matrix where each row is a sample and each column is a feature.
        y (np.ndarray):
            The target variable array, with each element corresponding to a sample in X.

    Returns:
        List[int]: The indices of the features in the best-performing set based on the merit score.

    """
    cache = _FeatureSelectorCache(X, y)
    n_features = X.shape[1]

    best_feature_set = max(
        ([*feature_set] for r in range(1, n_features + 1) for feature_set in combinations(range(n_features), r)),
        key=lambda feature_set: merit_calculation(tuple(feature_set), cache),
        default=[],
    )

    return best_feature_set


def _greedy_search(X: np.ndarray, y: np.ndarray) -> List[int]:
    """Performs a greedy forward search to find an optimal feature set based on merit score.

    This function implements a greedy forward search algorithm for feature selection, iteratively
    evaluating the impact of adding each feature on the overall merit score. It aims to find a
    locally optimal set of features by choosing the best single feature to add at each step.
    The search terminates when adding any feature does not improve the merit score, or if the
    merit score has not improved over the last five iterations, indicating a potential local
    maximum.

    Args:
        X (np.ndarray): The feature matrix where each row is a sample and each column is a feature.
        y (np.ndarray): The target variable array, with each element corresponding to a sample in X.

    Returns:
        List[int]: The indices of the features in the selected set based on the greedy forward search.

    """
    cache = _FeatureSelectorCache(X, y)
    n_features = X.shape[1]
    selected_features: List[int] = []
    merit_scores: List[float] = []

    while True:
        current_merit = merit_calculation(tuple(selected_features), cache)
        merit_scores.append(current_merit)
        add_candidates = [
            (merit_calculation(tuple(selected_features + [i]), cache), i)
            for i in range(n_features)
            if i not in selected_features
        ]
        best_merit, best_feature_to_add = max(add_candidates, key=lambda x: x[0], default=(-np.inf, -1))
        if best_merit <= current_merit:
            break
        selected_features.append(best_feature_to_add)

    return selected_features
