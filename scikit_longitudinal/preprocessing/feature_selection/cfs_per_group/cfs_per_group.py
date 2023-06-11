import re
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ray
from overrides import override

from scikit_longitudinal.preprocessing.feature_selection.cfs_per_group.utils import merit_calculation
from scikit_longitudinal.templates import CustomTransformerMixinEstimator


# pylint: disable=R0902
class CorrelationBasedFeatureSelectionPerGroup(CustomTransformerMixinEstimator):
    """Correlation-based Feature Selection (CFS) per group.

    This class performs feature selection using the correlation-based feature selection (CFS) algorithm on given data.
    The CFS algorithm is a filter method that selects features based on their correlation with the target variable and
    their mutual correlation with each other. This implementation supports the following search methods:
    exhaustiveSearch, or greedySearch. This implementation also supports a longitudinal component to handle feature
    selection for longitudinal data.

    Read more in the Notes below for implementation details.

    Args:
        non_longitudinal_features : Optional[List[int]] = None
            A list of feature indices that are not considered longitudinal. These features will not be affected by
            the longitudinal component of the algorithm.

        search_method : str, default="greedySearch"
            The search method to use. Options are "exhaustiveSearch", and "greedySearch".

        features_group : Optional[List[Tuple[int, ...]]], default=None
            A list of tuples of feature indices that represent the groups of features to be considered for feature
            selection per group (longitudinal component). If None, the CFS per group will not be used (no longitudinal
            component).

        parallel : bool, default=False
            Whether to use parallel processing for the CFS algorithm (especially useful for the exhaustive search method
            with the CFS per group, i.e. longitudinal component).

        cfs_longitudinal_outer_search_method : str, default=None
            The outer (to the final aggregated list of features) search method to use for the CFS per group
            (longitudinal component). If None, it defaults to the same as the `search_method`.

        cfs_longitudinal_inner_search_method : str, default="exhaustiveSearch"
            The inner (to each group) search method to use for the CFS per group (longitudinal component).

        cfs_per_group_version : str, default=2
            The version of the CFS per group algorithm to use. Options are "1" and "2". Version 2 is the improved with
            an outer search out of the final aggregated list of features of the first phase. Refer to the paper proposed
            below for more details.

        num_cpus : int, default=-1
            The number of CPUs to use for parallel processing. If -1, all available CPUs will be used.

    Attributes:
        selected_features_ : ndarray of shape (n_features,)
            The indices of the selected features.

    Examples:
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
        ...     search_method="greedySearch",
        ...     group_features=group_features,
        ...     parallel=True,
        ...     num_cpus=4
        ... )
        >>> X_longitudinal_selected = cfs_longitudinal.fit_transform(X, y)
        >>> X_longitudinal_selected.shape
        >>> # (100, N) ; N is the number of selected features

        >>> # Example of using the apply_selected_features_and_rename method:
        >>> data = np.random.random((100, 20))
        >>> df = pd.DataFrame(data, columns=[f'feature{i}_w1' for i in range(10)] +
            [f'feature{i}_w2' for i in range(10)])
        >>> y = np.random.randint(0, 2, 100)
        >>> non_longitudinal_features = [0, 1, 2]  # First three features are non-longitudinal
        >>> cfs = CorrelationBasedFeatureSelectionPerGroup(non_longitudinal_features=non_longitudinal_features)
        >>> cfs.fit(df, y)
        >>> df_selected = cfs.apply_selected_features_and_rename(df)
        >>> df_selected.columns
        >>> # Index([...]) ; Selected features and updated column names

    Notes:
        The improved CFS algorithm is based on the following references:

        * Zixiao. S. (2019, August 11). GitHub - ZixiaoShen
        /Correlation-based-Feature-Selection, available at:
        https://github.com/ZixiaoShen/Correlation-based-Feature-Selection

        The longitudinal component is based on the following paper and the original implementation, which is in JAVA
        that was reproduced in Python:

        * [VERSION-1 of the CFS Per Group] Pomsuwan, T. and Freitas, A.A., 2017, November.
          Feature selection for the classification of longitudinal human ageing data.
          In 2017 IEEE International Conference on Data Mining Workshops (ICDMW) (pp. 739-746). IEEE.

        * [VERSION-2 of the CFS Per Group] TODO Add the paper reference here

        * Pomsuwan, T. (2023, February 24). GitHub - mastervii/CSF_2-phase-variant, avaiable at:
          https://github.com/mastervii/CSF_2-phase-variant

    See also:
        * CustomTransformerMixinEstimator: Base class for all Transformer Mixin estimators in scikit-learn that we
        customed so that the original scikit-learn "check_x_y" is performed all the time.

    """

    # pylint: disable=too-many-arguments,invalid-name,signature-differs,no-member
    def __init__(
            self,
            non_longitudinal_features: Optional[List[int]] = None,
            search_method: str = "greedySearch",
            features_group: Optional[List[List[int]]] = None,
            parallel: bool = False,
            cfs_longitudinal_outer_search_method: str = None,
            cfs_longitudinal_inner_search_method: str = "exhaustiveSearch",
            cfs_per_group_version=2,
            num_cpus: int = -1,
            cfs_type: str = "cfs_longitudinal",
    ):
        assert search_method in {
            "exhaustiveSearch",
            "greedySearch",
        }, "search_method must be: 'exhaustiveSearch', or 'greedySearch'"

        self.search_method = search_method
        self.features_group = features_group
        self.parallel = parallel
        self.cfs_longitudinal_outer_search_method = (
            self.search_method if cfs_longitudinal_outer_search_method is None else cfs_longitudinal_outer_search_method
        )
        self.cfs_longitudinal_inner_search_method = cfs_longitudinal_inner_search_method
        self.selected_features_ = []
        self.selected_longitudinal_features_ = []
        self.non_longitudinal_features = non_longitudinal_features
        self.num_cpus = num_cpus
        if cfs_type == "auto":
            if self.features_group is not None:
                self.cfs_type = "cfs_longitudinal"
            else:
                self.cfs_type = "cfs"
        elif cfs_type in {"cfs", "cfs_longitudinal"}:
            self.cfs_type = cfs_type
        else:
            self.cfs_type = "cfs"
        self.cfs_per_group_version = cfs_per_group_version

    @override
    def _fit(self, X: np.ndarray, y: np.ndarray) -> "CorrelationBasedFeatureSelectionPerGroup":
        """Fits the CFS algorithm on the input data and target variable.

        This method applies the CFS algorithm to the input data, selecting the features that are correlated with the
        target variable, while having low mutual correlation with each other. It supports different search methods
        and an optional longitudinal component.

        Args:
            X (np.ndarray):
                The input data of shape (n_samples, n_features).
            y (np.ndarray):
                The target variable of shape (n_samples).

        Returns:
            CorrelationBasedFeatureSelectionPerGroup: The fitted instance of the CFS algorithm.

        """

        if self.features_group is not None:
            self.cfs_type = "cfs_longitudinal"

        if self.features_group is not None \
                and ray.is_initialized() is False \
                and self.parallel is True \
                and self.cfs_type != "cfs":
            if self.num_cpus != -1:
                ray.init(num_cpus=self.num_cpus)
            else:
                ray.init()

        if self.features_group is not None and self.cfs_type == "cfs_longitudinal":
            self.search_method = self.cfs_longitudinal_inner_search_method
            group_features_copy, group_selected_features = (
                (self.features_group.copy(), []) if self.features_group else ([], [])
            )
            self.features_group = None

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

            if self.cfs_per_group_version == 2:
                # Combine inner search results with non-longitudinal features
                combined_features = [
                                        index for sublist in group_selected_features for index in sublist
                                    ] + self.non_longitudinal_features

                # Run the outer search method on the final set of features
                # extracted from each group
                self.search_method = self.cfs_longitudinal_outer_search_method
                selected_indices = self._fit(X[:, combined_features], y).selected_features_

                # Get the final set of selected features based on the outer search results
                flattened_list = np.array(combined_features)
                self.selected_features_ = flattened_list[selected_indices].tolist()
            elif self.cfs_per_group_version == 1:
                # Only use the inner search results as the selected features for version 1
                flattened_list = np.array([index for sublist in group_selected_features for index in sublist])
                self.selected_features_ = (flattened_list.tolist() or []) + (self.non_longitudinal_features or [])
        else:
            match self.search_method:
                case "exhaustiveSearch":
                    self.selected_features_ = self._exhaustive_search(X, y)
                case "greedySearch":
                    self.selected_features_ = self._greedy_search(X, y)

        return self

    @override
    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Reduces the input data to only the selected features.

        Args:
            X:
                A numpy array of shape (n_samples, n_features) representing the input data.

        Returns:
            The reduced input data as a numpy array of shape (n_samples, n_selected_features).

        """
        if self.cfs_type == "cfs_longitudinal":
            return X
        return X[:, self.selected_features_]

    def _greedy_search(self, X: np.ndarray, y: np.ndarray) -> List[int]:
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

    def _exhaustive_search(self, X: np.ndarray, y: np.ndarray) -> List[int]:
        """Performs exhaustive search for feature selection.

        This method examines all possible combinations of features and selects
        the combination that has the highest merit score.

        Args:
            X (np.ndarray):
                Input data of shape (n_samples, n_features).
            y (np.ndarray):
                Target variable of shape (n_samples).

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

        Args:
            X (np.ndarray):
                The input data of shape (n_samples, n_features).
            y (np.ndarray):
                The target variable of shape (n_samples).
            group (Tuple[int]):
                A tuple of feature indices representing the group of features to fit the CFS algorithm on.

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

        Args:
            X (np.ndarray):
                The input data of shape (n_samples, n_features).
            y (np.ndarray):
                The target variable of shape (n_samples).
            group (Tuple[int]):
                A tuple of feature indices representing the group of features to fit the CFS algorithm on.

        Returns:
            List[int]: A list of selected feature indices for the given group.

        """
        return self._fit_subset(X, y, group)

    # pylint: disable=W9016
    @staticmethod
    def apply_selected_features_and_rename(
            df: pd.DataFrame, selected_features: List, regex_match=r"^(.+)_w(\d+)$"
    ) -> [pd.DataFrame, None]:
        """Apply selected features to the input DataFrame and rename non-longitudinal features.

        This function applies the selected features using the `selected_features_` attribute of the class.
        It also renames the non-longitudinal features that may have become non-longitudinal if only
        one wave remains after the feature selection process, to avoid them being considered as
        longitudinal attributes during future automatic feature grouping.

        Args:
            df : pd.DataFrame
                The input DataFrame to apply the selected features and perform renaming.
            selected_features : List
                The list of selected features to apply to the input DataFrame.
            regex_match : str
                The regex pattern to use for renaming non-longitudinal features. Follow by default the
                Elsa naming convention for longitudinal features.

        Returns
        -------
        pd.DataFrame
            The modified DataFrame with selected features applied and non-longitudinal features renamed.

        """
        # Apply selected features
        if selected_features:
            df = df.iloc[:, selected_features].copy()

        # Rename non-longitudinal features
        non_longitudinal_features: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for col in df.columns:
            if not isinstance(col, str):
                continue
            if match := re.match(regex_match, col):
                feature_base_name, wave_number = match.groups()
                non_longitudinal_features[feature_base_name].append((col, wave_number))

        for base_name, columns in non_longitudinal_features.items():
            if len(columns) == 1:
                old_name, wave_number = columns[0]
                new_name = f"{base_name}_wave{wave_number}"
                df.rename(columns={old_name: new_name}, inplace=True)
        return df
