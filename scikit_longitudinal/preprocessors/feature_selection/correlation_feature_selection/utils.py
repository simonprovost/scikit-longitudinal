# flake8: noqa
# pylint: skip-file
from itertools import combinations
from math import log
from typing import List

import numpy as np


def discrete_entropy(samples, base=2):
    """Calculate the discrete entropy of a list of samples, where each sample can be any hashable object.

    Notes:
        This function computes a discrete entropy estimator given a list of samples which can be any hashable
        object.

    Args:
        samples:
            A list of hashable objects.
        base:
            An integer that represents the logarithmic base to use in the entropy calculation.

    Returns:
        A float that represents the discrete entropy.

    """

    return entropy_from_probabilities(histogram(samples), base=base)


def discrete_mutual_information(x, y):
    """Calculate the discrete mutual information of two lists of samples, where each sample can be any hashable object.

    Notes:
        This function computes a discrete mutual information estimator given a list of samples which can be any
        hashable object.

    Args:
        x:
            A list of hashable objects.
        y:
            A list of hashable objects.

    Returns:
        A float that represents the discrete mutual information.

    """

    return -discrete_entropy(list(zip(x, y))) + discrete_entropy(x) + discrete_entropy(y)


def histogram(samples):
    """Compute a histogram from a list of samples.

    Args:
        samples:
            A list of hashable objects.

    Returns:
        A list of floats that represents the frequencies of each sample in the input list.

    """
    frequencies = {}
    for sample in samples:
        frequencies[sample] = frequencies.get(sample, 0) + 1
    return map(lambda freq: float(freq) / len(samples), frequencies.values())


def entropy_from_probabilities(probabilities, base=2):
    """Compute the entropy from a list of normalized probabilities.

    Notes:
        This function computes the entropy (base 2) from a list of normalized probabilities.
    Args:
        probabilities:
            A list of floats that represents the probabilities.
        base:
            An integer that represents the logarithmic base to use in the entropy calculation.

    Returns:
        A float that represents the entropy.

    """
    return -sum(map(elementwise_log, probabilities)) / log(base)


def elementwise_log(x):
    """Compute the element-wise logarithm for entropy calculation.

    Args:
        x:
            A float.

    Returns:
        A float that represents the element-wise logarithm.

    """
    return 0 if x <= 0.0 or x >= 1.0 else x * log(x)


def information_gain(feature1, feature2):
    """Calculate the information gain between two lists of samples.

    Args:
        feature1:
            A numpy array of shape (n_samples,).
        feature2:
            A numpy array of shape (n_samples,).

    Returns:
        The information gain as a float.

    """

    return discrete_entropy(feature1) - conditional_entropy(feature1, feature2)


def conditional_entropy(feature1, feature2):
    """Calculate the conditional entropy between two lists of samples.

    Args:
        feature1:
            A numpy array of shape (n_samples,).
        feature2:
            A numpy array of shape (n_samples,).

    Returns:
        The conditional entropy as a float.

    """

    return discrete_entropy(feature1) - discrete_mutual_information(feature1, feature2)


def symmetrical_uncertainty(feature1, feature2):
    """Calculate the symmetrical uncertainty between two lists of samples.

    Args:
        feature1:
            A numpy array of shape (n_samples,).
        feature2:
            A numpy array of shape (n_samples,).

    Returns:
        The symmetrical uncertainty as a float.

    """
    # calculate information gain of f1 and f2, t1 = ig(f1, f2)
    t1 = information_gain(feature1, feature2)
    # calculate entropy of f1
    t2 = discrete_entropy(feature1)
    # calculate entropy of f2
    t3 = discrete_entropy(feature2)

    return 2.0 * t1 / (t2 + t3)


def merit_calculation(X: np.ndarray, y: np.ndarray) -> float:
    """Calculate the merit of a feature subset X given class labels y.

    Args:
        X:
            A numpy array of shape (n_samples, n_features) representing the input data.
        y:
            A numpy array of shape (n_samples) representing the input class labels.

    Returns:
        The merit of the feature subset X as a float.

    """

    _, n_features = X.shape
    feature_feature_correlation_sum = 0
    class_feature_correlation_sum = 0

    for i in range(n_features):
        feature_i = X[:, i]
        class_feature_correlation_sum += symmetrical_uncertainty(feature_i, y)
        for j in range(n_features):
            if j > i:
                feature_j = X[:, j]
                feature_feature_correlation_sum += symmetrical_uncertainty(feature_i, feature_j)

    feature_feature_correlation_sum *= 2
    denominator = np.sqrt(n_features + feature_feature_correlation_sum)

    return 0 if denominator == 0 else class_feature_correlation_sum / denominator


def _exhaustive_search(X: np.ndarray, y: np.ndarray) -> List[int]:
    n_features = X.shape[1]
    return max(
        (feature_set for r in range(1, n_features + 1) for feature_set in combinations(range(n_features), r)),
        key=lambda feature_set: merit_calculation(X[:, feature_set], y),
        default=[],
    )


def _greedy_search(X: np.ndarray, y: np.ndarray) -> List[int]:
    """Performs greedy search for feature selection.

     This method starts with an empty set of features and iteratively adds
     or removes a feature based on the merit score, which takes into account
     the correlation with the target variable and the inter-feature correlations.
     The process continues until no further improvement in the merit score is achieved.

    Args:
        X (np.ndarray):
            Input data of shape (n_samples, n_features).
        y (np.ndarray):
            Target variable of shape (n_samples).

    Returns:
        List[int]: A list of selected feature indices.

    """
    n_features = X.shape[1]
    selected_features: List[int] = []
    merit_scores: List[float] = []

    while True:
        current_merit = merit_calculation(X[:, selected_features], y)
        merit_scores.append(current_merit)

        add_candidates = [
            (merit_calculation(X[:, selected_features + [i]], y), i)
            for i in range(n_features)
            if i not in selected_features
        ]
        remove_candidates = [
            (merit_calculation(X[:, [i for i in selected_features if i != j]], y), j) for j in selected_features
        ]

        all_candidates = add_candidates + remove_candidates
        best_merit, best_idx = max(all_candidates, key=lambda x: x[0], default=(-np.inf, -1))

        if len(merit_scores) > 5:
            if all(merit_scores[-(i + 1)] <= merit_scores[-(i + 2)] for i in range(5)):
                break

        if best_merit <= current_merit:
            break

        if best_idx in selected_features:
            selected_features.remove(best_idx)
        else:
            selected_features.append(best_idx)
    return selected_features
