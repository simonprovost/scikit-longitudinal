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
class CorrelationBasedFeatureSelection(CustomTransformerMixinEstimator):
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
            cfs_type: str = "auto",
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
                self.cfs_type_ = "cfs_longitudinal"
            else:
                self.cfs_type_ = "cfs"
        elif cfs_type in {"cfs", "cfs_longitudinal"}:
            self.cfs_type_ = cfs_type
        else:
            self.cfs_type_ = "cfs"
        self.version = cfs_per_group_version

        if self.features_group is not None \
                and ray.is_initialized() is False \
                and self.parallel is True \
                and self.cfs_type_ != "cfs":
            if num_cpus != -1:
                ray.init(num_cpus=num_cpus)
            else:
                ray.init()

    @override
    def _fit(self, X: np.ndarray, y: np.ndarray) -> "CorrelationBasedFeatureSelectionPerGroup":
        if self.features_group is not None and self.cfs_type_ == "cfs_longitudinal":
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

            if self.version == 2:
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
            elif self.version == 1:
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
        if self.cfs_type_ == "cfs_longitudinal":
            return X
        return X[:, self.selected_features_]

    def _greedy_search(self, X: np.ndarray, y: np.ndarray) -> List[int]:
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
        n_features = X.shape[1]
        return max(
            (feature_set for r in range(1, n_features + 1) for feature_set in combinations(range(n_features), r)),
            key=lambda feature_set: merit_calculation(X[:, feature_set], y),
            default=[],
        )

    def _fit_subset(self, X: np.ndarray, y: np.ndarray, group: Tuple[int]) -> List[int]:
        X_group = X[:, group]
        self._fit(X_group, y)
        return [group[i] for i in self.selected_features_]

    @ray.remote
    def _ray_fit_subset(self, X: np.ndarray, y: np.ndarray, group: Tuple[int]) -> List[int]:
        return self._fit_subset(X, y, group)

    # pylint: disable=W9016
    @staticmethod
    def apply_selected_features_and_rename(
            df: pd.DataFrame, selected_features: List, regex_match=r"^(.+)_w(\d+)$"
    ) -> [pd.DataFrame, None]:
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
