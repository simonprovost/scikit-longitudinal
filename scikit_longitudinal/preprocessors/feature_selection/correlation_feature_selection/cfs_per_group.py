import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ray
from overrides import override

from scikit_longitudinal.preprocessors.feature_selection.correlation_feature_selection.algorithms import (
    _exhaustive_search,
    _greedy_search,
)
from scikit_longitudinal.templates import CustomTransformerMixinEstimator


# pylint: disable=R0902, R0801, R0912, W0511
class CorrelationBasedFeatureSelectionPerGroup(CustomTransformerMixinEstimator):
    """Correlation-based Feature Selection (CFS) per group (CFS Per Group).

    ⚠️ Scikit-Longitudinal's docstrings will be updated to reflect the most recent documentation available on Github.
    If something is inconsistent, consult the documentation first, then file an issue. ⚠️

    This class performs feature selection using the correlation-based feature selection (CFS) algorithm on given data.
    The CFS algorithm is a filter method that selects features based on their correlation with the target variable and
    their mutual correlation with each other. This implementation supports the following search methods:
    exhaustiveSearch, or greedySearch. This implementation concern the support for the longitudinal component to
    handle feature selection for longitudinal data. Hence, the CFS per group. If no longitudinal component is needed,
    refer to the CorrelationBasedFeatureSelection class.

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

        outer_search_method : str, default=None
            The outer (to the final aggregated list of features) search method to use for the CFS per group
            (longitudinal component). If None, it defaults to the same as the `search_method`.

        inner_search_method : str, default="exhaustiveSearch"
            The inner (to each group) search method to use for the CFS per group (longitudinal component).

        version : str, default=2
            The version of the CFS per group algorithm to use. Options are "1" and "2". Version 2 is the improved with
            an outer search out of the final aggregated list of features of the first phase. Refer to the paper proposed
            below for more details.

        num_cpus : int, default=-1
            The number of CPUs to use for parallel processing. If -1, all available CPUs will be used.

    Attributes:
        selected_features_ : ndarray of shape (n_features,)
            The indices of the selected features.

    Examples:
        >>> # With the longitudinal component:
        >>> features_group = [(0, 1, 2), (3, 4, 5), (6, 7, 8, 9)]
        >>> cfs_longitudinal = CorrelationBasedFeatureSelectionPerGroup(
        ...     features_group=features_group
        ... )
        >>> cfs_longitudinal.fit(X, y)
        >>> X_longitudinal_selected = cfs_longitudinal.transform(X)
        >>> X_longitudinal_selected.shape
        >>> # (100, N) ; N is the number of selected features

        >>> # With the longitudinal component and parallel processing:
        >>> features_group = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14), (15, 16, 17), (18, 19)]
        >>> cfs_longitudinal  = CorrelationBasedFeatureSelectionPerGroup(
        ...     search_method="greedySearch",
        ...     features_group=features_group,
        ...     parallel=True,
        ...     num_cpus=4
        ... )
        >>> X_longitudinal_selected = cfs_longitudinal.fit_transform(X, y)
        >>> X_longitudinal_selected.shape
        >>> # (100, N) ; N is the number of selected features

        >>> # Example of using the apply_selected_features_and_rename method (alternative to transform):
        >>> data = np.random.random((100, 20))
        >>> df = pd.DataFrame(data, columns=[f'feature{i}_w1' for i in range(10)] +
        ...     [f'feature{i}_w2' for i in range(10)]
        ... )
        >>> y = np.random.randint(0, 2, 100)
        >>> non_longitudinal_features = [0, 1, 2]  # First three features are non-longitudinal
        >>> cfs = CorrelationBasedFeatureSelectionPerGroup(
        ...     # features_group=<your_features_group>,
        ...     non_longitudinal_features=non_longitudinal_features
        ... )
        >>> cfs.fit(df, y)
        >>> df_selected = cfs.apply_selected_features_and_rename(df)
        >>> df_selected.columns
        >>> # Index([...]) ; Selected features and ** updated column names **

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

        * [VERSION-2 of the CFS Per Group] T. Pomsuwan and A. Freitas, “Feature selection for the classification of
          longitudinal human ageing data,” Master’s thesis, University of Kent, Feb 2018. [Online]. Available:
          https://kar.kent.ac.uk/66568/

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
        outer_search_method: str = None,
        inner_search_method: str = "exhaustiveSearch",
        version=1,
        num_cpus: int = -1,
    ):
        assert search_method in {
            "exhaustiveSearch",
            "greedySearch",
        }, "search_method must be: 'exhaustiveSearch', or 'greedySearch'"

        self.search_method = search_method
        self.features_group = features_group
        self.parallel = parallel
        self.outer_search_method = self.search_method if outer_search_method is None else outer_search_method
        self.inner_search_method = inner_search_method
        self.non_longitudinal_features = non_longitudinal_features
        self.num_cpus = num_cpus
        self.version = version
        self.selected_features_ = []
        self.selected_longitudinal_features_ = []

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
        if self.features_group is not None and ray.is_initialized() is False and self.parallel is True:
            if self.num_cpus != -1:
                ray.init(num_cpus=self.num_cpus)
            else:
                ray.init()

        # TODO: Make sure to rework the too many branches warning
        if self.features_group is not None:
            self.search_method = self.inner_search_method
            group_features_copy, group_selected_features = (
                (self.features_group.copy(), []) if self.features_group else ([], [])
            )

            self.features_group = None

            if self.parallel:
                futures = [self._ray_fit_subset.remote(self, X, y, group) for group in group_features_copy]
                while futures:
                    ready_futures, remaining_futures = ray.wait(futures)
                    result = ray.get(ready_futures[0])
                    group_selected_features.append(result)
                    futures = remaining_futures
            else:
                group_selected_features = [self._fit_subset(X, y, group) for group in group_features_copy]

            if self.version == 2:
                combined_features = [index for sublist in group_selected_features for index in sublist] + (
                    self.non_longitudinal_features or []
                )
                self.search_method = self.outer_search_method
                selected_indices = self._fit(X[:, combined_features], y).selected_features_
                flattened_list = np.array(combined_features)
                self.selected_features_ = flattened_list[selected_indices].tolist()
            elif self.version == 1:
                flattened_list = np.array([index for sublist in group_selected_features for index in sublist])
                self.selected_features_ = (flattened_list.tolist() or []) + (self.non_longitudinal_features or [])
            else:
                raise ValueError(f"Version {self.version} is not supported. Please choose version 1 or 2.")
        else:
            if self.search_method == "exhaustiveSearch":
                self.selected_features_ = _exhaustive_search(X, y)
            elif self.search_method == "greedySearch":
                self.selected_features_ = _greedy_search(X, y)
            else:
                raise ValueError(f"Search method {self.search_method} is not supported.")

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
        return X

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

        Returns:
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
