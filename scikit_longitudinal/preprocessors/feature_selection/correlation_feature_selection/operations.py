# flake8: noqa
# pylint: skip-file
from functools import lru_cache
from math import sqrt
from typing import Tuple

import numpy as np
from scipy.stats import pointbiserialr


class _FeatureSelectorCache:
    """Cache for feature correlations to speed up feature selection processes.

    This class is designed to cache the correlation values between features and between
    features and the class variable. It utilises Python's `lru_cache` decorator to memorise
    previously computed correlation values, thus reducing the computational cost of repeated
    calculations during feature selection.

    Args:
        X (np.ndarray):
            The feature matrix.
        y (np.ndarray):
            The target variable array.

    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialises the cache with feature matrix and target variable.
        """
        self.X = X
        self.y = y

    @lru_cache(maxsize=None)
    def feature_class_correlation(self, feature_index: int) -> float:
        """Calculates the absolute correlation between a given feature and the class variable.

        This method computes the point-biserial correlation coefficient between a given feature
        and the class variable. The absolute value of the correlation is returned to consider
        both positive and negative relationships.

        Args:
            feature_index (int): Index of the feature in the feature matrix.

        Returns:
            float: The absolute value of the correlation coefficient between the feature
            and the class variable.

        """
        correlation, _ = pointbiserialr(self.X[:, feature_index], self.y)
        return abs(correlation)

    @lru_cache(maxsize=None)
    def feature_feature_correlation(self, feature_index_1: int, feature_index_2: int) -> float:
        """Calculates the absolute correlation between two features.

        Computes the Pearson correlation coefficient between two features identified by their
        indices. The absolute value of the correlation is returned to capture both types of
        linear relationships.

        Args:
            feature_index_1 (int): Index of the first feature in the feature matrix.
            feature_index_2 (int): Index of the second feature in the feature matrix.

        Returns:
            float: The absolute value of the Pearson correlation coefficient between
            the two specified features.

        """
        correlation = np.corrcoef(self.X[:, feature_index_1], self.X[:, feature_index_2])[0, 1]
        return abs(correlation)


def merit_calculation(feature_indices: Tuple[int], cache: _FeatureSelectorCache) -> float:
    """Calculates the merit of a set of features using cached correlations.

    This function computes the merit of a given set of features based on the average
    feature-to-class correlation and the average feature-to-feature correlation, using cached
    values to fasten processes. The merit is a measure used in feature selection to
    evaluate the potential effectiveness of a feature set in classification tasks.

    Args:
        feature_indices (Tuple[int]):
            Indices of the features in the set.
        cache (_FeatureSelectorCache):
            An instance of _FeatureSelectorCache containing precomputed correlation values.

    Returns:
        float: The calculated merit of the given feature set.

    """
    feature_to_class_correlations = [cache.feature_class_correlation(index) for index in feature_indices]
    avg_feature_to_class_correlation = np.mean(feature_to_class_correlations) if feature_to_class_correlations else 0

    feature_to_feature_correlations = [
        cache.feature_feature_correlation(feature_indices[i], feature_indices[j])
        for i in range(len(feature_indices))
        for j in range(i + 1, len(feature_indices))
    ]
    avg_feature_to_feature_correlation = (
        np.mean(feature_to_feature_correlations) if feature_to_feature_correlations else 0
    )

    total_features = len(feature_indices)
    if total_features == 0:
        return 0
    merit_score = (total_features * avg_feature_to_class_correlation) / sqrt(
        total_features + total_features * (total_features - 1) * avg_feature_to_feature_correlation
    )
    return merit_score
