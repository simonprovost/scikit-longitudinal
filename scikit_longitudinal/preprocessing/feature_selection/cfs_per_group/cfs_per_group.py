from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
import ray
from overrides import override

from scikit_longitudinal.preprocessing.feature_selection.cfs_per_group.utils import symmetrical_uncertainty
from scikit_longitudinal.templates import CustomBaseEstimator


class CorrelationBasedFeatureSelectionPerGroup(CustomBaseEstimator):
    """Correlation-based Feature Selection (CFS) per group.

    This class performs feature selection using the correlation-based feature selection (CFS) algorithm on given data.
    The CFS algorithm is a filter method that selects features based on their correlation with the target variable and
    their mutual correlation with each other. This implementation supports the following search methods:
    forwardBestFirstSearch, exhaustiveSearch, backwardBestFirstSearch, bidirectionalSearch, or greedySearch. This
    implementation also supports a longitudinal component to handle feature selection for longitudinal data.

    Read more in the Notes below for implementation details.

    Parameters
    ----------
    search_method : str, default="forwardBestFirstSearch"
        The search method to use. Options are "forwardBestFirstSearch", "exhaustiveSearch", "backwardBestFirstSearch",
        "bidirectionalSearch", and "greedySearch".

    consecutive_non_improving_subsets_limit : int, default=5
        The maximum number of consecutive non-improving subsets to allow for the original implementation of the
        forward-best-first search method.

    group_features : List[Tuple[int, ...]], default=None
        A list of tuples of feature indices that represent the groups of features to be considered for feature
        selection per group (longitudinal component). If None, the CFS per group will not be used (no longitudinal
        component).

    parallel : bool, default=False
        Whether to use parallel processing for the CFS algorithm (especially useful for the exhaustive search method
        with the CFS per group, i.e. longitudinal component).

    cfs_longitudinal_outer_search_method : str, default=None
        The outer (to the final aggregated list of features) search method to use for the CFS per group (longitudinal
        component). If None, it defaults to the same as the `search_method`.

    cfs_longitudinal_inner_search_method : str, default="exhaustiveSearch"
        The inner (to each group) search method to use for the CFS per group (longitudinal component).

    num_cpus : int, default=-1
        The number of CPUs to use for parallel processing. If -1, all available CPUs will be used.

    Attributes
    ----------
    selected_features_ : ndarray of shape (n_features,)
        The indices of the selected features.

    Examples
    ----------

    >>>  # Without the longitudinal component (original CFS):
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from scikit_longitudinal import CorrelationBasedFeatureSelectionPerGroup
    >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    >>> cfs = CorrelationBasedFeatureSelectionPerGroup()
    >>> cfs.fit(X, y)
    >>> X_selected = cfs.transform(X)
    >>> X_selected.shape
    >>> # (100, N) ; N is the number of selected features

    >>> # With the longitudinal component:
    >>> group_features = [(0, 1, 2), (3, 4, 5), (6, 7, 8, 9)]
    >>> cfs_longitudinal = CorrelationBasedFeatureSelectionPerGroup(
    ...     group_features=group_features)
    >>> cfs_longitudinal.fit(X, y)
    >>> X_longitudinal_selected = cfs_longitudinal.transform(X)
    >>> X_longitudinal_selected.shape
    >>> # (100, N) ; N is the number of selected features

    >>> # With the longitudinal component and parallel processing:
    >>> group_features = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14), (15, 16, 17), (18, 19)]
    >>> cfs_longitudinal  = CorrelationBasedFeatureSelectionPerGroup(
    ...     search_method="forwardBestFirstSearch",
    ...     group_features=group_features,
    ...     parallel=True,
    ...     num_cpus=4
    ... )
    >>> X_longitudinal_selected = cfs_longitudinal.fit_transform(X, y)
    >>> X_longitudinal_selected.shape
    >>> # (100, N) ; N is the number of selected features


    Notes
    ----------

    The improved CFS algorithm is based on the following references:

    * Zixiao. S. (2019, August 11). GitHub - ZixiaoShen
    /Correlation-based-Feature-Selection, avaiable at:
    https://github.com/ZixiaoShen/Correlation-based-Feature-Selection

    The longitudinal component is based on the following paper and the original implementation, which is in JAVA
    that was reproduced in Python:

    * Pomsuwan, T. and Freitas, A.A., 2017, November.
      Feature selection for the classification of longitudinal human ageing data.
      In 2017 IEEE International Conference on Data Mining Workshops (ICDMW) (pp. 739-746). IEEE.

    * Pomsuwan, T. (2023, February 24). GitHub - mastervii/CSF_2-phase-variant, avaiable at:
      https://github.com/mastervii/CSF_2-phase-variant


    See also
    ----------

    * CustomBaseEstimator: Base class for all estimators in scikit-learn that we customed so that the original
      scikit-learn "check_x_y" is performed all the time.


    Methods
    ----------
    fit(X, y)
        Fit the CFS algorithm to the input data and target variable.

    transform(X)
        Reduce the input data to only the selected features.

    fit_transform(X, y)
        Fit the CFS algorithm and return the reduced input data.
    """

    # pylint: disable=too-many-arguments,invalid-name,signature-differs,no-member
    def __init__(
            self,
            search_method: str = "forwardBestFirstSearch",
            consecutive_non_improving_subsets_limit: int = 5,
            group_features: Optional[List[Tuple[int, ...]]] = None,
            parallel: bool = False,
            cfs_longitudinal_outer_search_method: str = None,
            cfs_longitudinal_inner_search_method: str = "exhaustiveSearch",
            num_cpus: int = -1,
    ):
        assert search_method in {
            "forwardBestFirstSearch",
            "exhaustiveSearch",
            "backwardBestFirstSearch",
            "bidirectionalSearch",
            "greedySearch",
        }, (
            "search_method must be 'forwardBestFirstSearch', 'exhaustiveSearch', 'backwardBestFirstSearch', "
            "'bidirectionalSearch', or 'greedySearch'"
        )

        if group_features is not None and ray.is_initialized() is False:
            if num_cpus != -1:
                ray.init(num_cpus=num_cpus)
            else:
                ray.init()
        self.search_method = search_method
        self.consecutive_non_improving_subsets_limit = consecutive_non_improving_subsets_limit
        self.longitudinal_group_features = group_features
        self.parallel = parallel
        self.cfs_longitudinal_outer_search_method = (
            self.search_method if cfs_longitudinal_outer_search_method is None else cfs_longitudinal_outer_search_method
        )
        self.cfs_longitudinal_inner_search_method = cfs_longitudinal_inner_search_method
        self.selected_features_ = []

    def _greedy_search(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        """Performs greedy search for feature selection.

         This method starts with an empty set of features and iteratively adds
         or removes a feature based on the merit score, which takes into account
         the correlation with the target variable and the inter-feature correlations.
         The process continues until no further improvement in the merit score is achieved.


        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray): Target variable of shape (n_samples).

        Returns:
            List[int]: A list of selected feature indices.
        """
        n_features = X.shape[1]
        selected_features: List[int] = []

        while True:
            current_merit = merit_calculation(X[:, selected_features], y)

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

            if best_merit <= current_merit:
                break

            if best_idx in selected_features:
                selected_features.remove(best_idx)
            else:
                selected_features.append(best_idx)
        return selected_features

    def _bidirectional_search(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        """Performs bidirectional search for feature selection.

         This method simultaneously performs forward and backward searches.
         The forward search starts with an empty set of features and iteratively adds
         the feature with the highest merit score. The backward search starts with a
         full set of features and iteratively removes the feature with the lowest
         merit score. The search stops when the forward and backward feature sets are equal.


        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray): Target variable of shape (n_samples).

        Returns:
            List[int]: A list of selected feature indices.
        """
        n_features = X.shape[1]
        forward_features: List[int] = []
        backward_features = list(range(n_features))

        while True:
            forward_merit, forward_idx = max(
                (
                    (merit_calculation(X[:, forward_features + [i]], y), i)
                    for i in range(n_features)
                    if i not in forward_features
                ),
                key=lambda x: x[0],
                default=(-np.inf, -1),
            )

            backward_merit, backward_idx = min(
                ((merit_calculation(X[:, [i for i in backward_features if i != j]], y), j) for j in backward_features),
                key=lambda x: x[0],
                default=(np.inf, -1),
            )

            if forward_merit > backward_merit:
                forward_features.append(forward_idx)
            else:
                backward_features.remove(backward_idx)

            if set(forward_features) == set(backward_features):
                break

        return forward_features

    def _backward_best_first_search(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        """Performs backward best-first search for feature selection.

        This method starts with a set of all features and iteratively removes
        the feature that has the lowest correlation with the target variable
        until the merit criterion starts decreasing.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray): Target variable of shape (n_samples).

        Returns:
            List[int]: A list of selected feature indices.
        """
        n_features = X.shape[1]
        selected_features = list(range(n_features))
        merit_values: List[float] = []

        def merit_decreasing(merits: List[float]) -> bool:
            """Checks if the merit is decreasing.

            This method checks if the merit is decreasing by comparing the last two
            merit values in the list. If the list has less than two elements, it
            returns False. Otherwise, it returns True if the last element is less
            than the second last element, False otherwise.

            Args:
                merits (List[float]): A list of merit values.

            Returns:
                bool: True if the merit is decreasing, False otherwise.
            """
            return False if len(merits) <= 1 else merits[-1] < merits[-2]

        while True:
            merit, idx = min(
                ((merit_calculation(X[:, [i for i in selected_features if i != j]], y), j) for j in selected_features),
                key=lambda x: x[0],
                default=(np.inf, -1),
            )
            selected_features.remove(idx)
            merit_values.append(merit)

            if merit_decreasing(merit_values):
                break

        return selected_features

    def _forward_best_first_search(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        """Performs forward best-first search for feature selection.

         This method starts with an empty set of features and iteratively adds
         the feature that maximizes the merit score, considering the correlation
         with the target variable and the inter-feature correlations.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray): Target variable of shape (n_samples).

        Returns:
            List[int]: A list of selected feature indices.
        """
        n_features = X.shape[1]
        selected_features: List[int] = []
        merit_values: List[float] = []

        def non_improving_subsets_reached(merits: List[float], limit: int) -> bool:
            """
            Determine if the maximum number of consecutive non-improving merit values has been reached.

            This function checks if the merit values in the `merits` list are not improving for the last `limit`
            number of elements. If the merit values are not improving or equal, it returns True, otherwise False.

            Notes:
                This function comes from the original implementation's concept.

            Args:
                merits (List[float]): A list of merit values for different feature subsets.
                limit (int): The maximum number of consecutive non-improving merit values allowed.

            Returns: bool: True if the maximum number of consecutive non-improving merit values has been reached,
            False otherwise.
            """
            if len(merits) <= limit:
                return False
            return all(merits[i] <= merits[i - 1] for i in range(-1, -limit - 1, -1))

        while True:
            merit, idx = max(
                (
                    (merit_calculation(X[:, selected_features + [i]], y), i)
                    for i in range(n_features)
                    if i not in selected_features
                ),
                key=lambda x: x[0],
                default=(-np.inf, -1),
            )
            selected_features.append(idx)
            merit_values.append(merit)

            if non_improving_subsets_reached(merit_values, self.consecutive_non_improving_subsets_limit):
                break

        return selected_features

    def _exhaustive_search(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        """Performs exhaustive search for feature selection.

        This method examines all possible combinations of features and selects
        the combination that has the highest merit score.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray): Target variable of shape (n_samples).

        Returns:
            List[int]: A list of selected feature indices.
        """
        n_features = X.shape[1]
        return max(
            (feature_set for r in range(1, n_features + 1) for feature_set in combinations(range(n_features), r)),
            key=lambda feature_set: merit_calculation(X[:, feature_set], y),
            default=[],
        )

    def _fit_subset(self, X: np.ndarray, y: np.ndarray, group: Tuple[int]) -> List[int]:
        """Fits the CFS algorithm on a subset of the input data specified by the group.

        This method applies the CFS algorithm to a specific group of features in the input data.
        It is called during the computation of the longitudinal component of the CFS algorithm.

        Args: X (np.ndarray): The input data of shape (n_samples, n_features). y (np.ndarray): The target variable of
        shape (n_samples). group (Tuple[int]): A tuple of feature indices representing the group of features to fit
        the CFS algorithm on.

        Returns:
            List[int]: A list of selected feature indices for the given group.
        """

        X_group = X[:, group]
        self._fit(X_group, y)
        return [group[i] for i in self.selected_features_]

    @ray.remote
    def _ray_fit_subset(self, X: np.ndarray, y: np.ndarray, group: Tuple[int]) -> List[int]:
        """Ray remote function for fitting the CFS algorithm on a subset of the input data specified by the group.

        This method applies the CFS algorithm to a specific group of features in the input data, using Ray for
        parallel computation. It is called during the computation of the longitudinal component of the CFS algorithm
        when parallel processing is enabled.

        Args: X (np.ndarray): The input data of shape (n_samples, n_features). y (np.ndarray): The target variable of
        shape (n_samples). group (Tuple[int]): A tuple of feature indices representing the group of features to fit
        the CFS algorithm on.

        Returns:
            List[int]: A list of selected feature indices for the given group.
        """
        return self._fit_subset(X, y, group)

    @override
    def _fit(self, X: np.ndarray, y: np.ndarray) -> "CorrelationBasedFeatureSelectionPerGroup":
        """Fits the CFS algorithm on the input data and target variable.

        This method applies the CFS algorithm to the input data, selecting the features that are correlated with the
        target variable, while having low mutual correlation with each other. It supports different search methods
        and an optional longitudinal component.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features).
            y (np.ndarray): The target variable of shape (n_samples).

        Returns:
            CorrelationBasedFeatureSelectionPerGroup: The fitted instance of the CFS algorithm.
        """
        if self.longitudinal_group_features is not None:
            self.search_method = self.cfs_longitudinal_inner_search_method
            group_features_copy, group_selected_features = (
                (self.longitudinal_group_features.copy(), []) if self.longitudinal_group_features else ([], [])
            )
            self.longitudinal_group_features = None

            # Run the inner search method for each group of features in
            # parallel or not
            if self.parallel:
                futures = [self._ray_fit_subset.remote(self, X, y, group) for group in group_features_copy]
                while futures:
                    ready_futures, remaining_futures = ray.wait(futures)
                    result = ray.get(ready_futures[0])
                    group_selected_features.append(result)
                    futures = remaining_futures
            else:
                group_selected_features = [self._fit_subset(X, y, group) for group in group_features_copy]

            # Run the outer search method on the final set of features
            # extracted from each group
            self.search_method = self.cfs_longitudinal_outer_search_method
            self._fit(X[:, [index - 1 for sublist in group_selected_features for index in sublist]], y)
        else:
            match self.search_method:
                case "forwardBestFirstSearch":
                    self.selected_features_ = self._forward_best_first_search(X, y)
                case "exhaustiveSearch":
                    self.selected_features_ = self._exhaustive_search(X, y)
                case "backwardBestFirstSearch":
                    self.selected_features_ = self._backward_best_first_search(X, y)
                case "bidirectionalSearch":
                    self.selected_features_ = self._bidirectional_search(X, y)
                case "greedySearch":
                    self.selected_features_ = self._greedy_search(X, y)

        return self

    @override
    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Reduces the input data to only the selected features.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.

        Returns:
            The reduced input data as a numpy array of shape (n_samples, n_selected_features).
        """
        return X[:, self.selected_features_]


def merit_calculation(X: np.ndarray, y: np.ndarray) -> float:
    """Calculate the merit of a feature subset X given class labels y.

    Args:
        X: A numpy array of shape (n_samples, n_features) representing the input data.
        y: A numpy array of shape (n_samples) representing the input class labels.

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
    return class_feature_correlation_sum / np.sqrt(n_features + feature_feature_correlation_sum)
