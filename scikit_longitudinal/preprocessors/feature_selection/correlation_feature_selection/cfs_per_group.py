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

    The `CorrelationBasedFeatureSelectionPerGroup` class implements the CFS-Per-Group algorithm, a longitudinal variant
    of the standard CFS method. It is designed to handle feature selection in longitudinal datasets by considering
    temporal variations across multiple waves (time points). The algorithm operates in two phases:

    1. **Phase 1**: For each longitudinal feature group, CFS with a specified search method (e.g., exhaustive or greedy)
       is applied to select relevant and non-redundant features across waves. The selected features are then aggregated.
    2. **Phase 2**: The aggregated features from Phase 1 are combined with non-longitudinal features, and a standard CFS
       is applied to further refine the selection by removing redundant features.

    !!! quote "CFS-Per-Group: A Longitudinal Variation of CFS"
        CFS-Per-Group, also known as `Exh-CFS-Gr` in the literature, adapts the standard CFS method to longitudinal data.
        It is particularly useful for datasets where features are collected over multiple time points, such as in ageing
        studies or health monitoring.

        For scientific references, see the Notes section below.

    !!! note "Standard CFS Implementation"
        For the standard CFS algorithm without the longitudinal component, refer to the `CorrelationBasedFeatureSelection`
        class.

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
        non_longitudinal_features (Optional[List[int]], optional): List of indices for non-longitudinal features.
            These features are not part of the temporal matrix and are treated separately. Defaults to None.
        search_method (str, optional): The search method for Phase 1. Options are "exhaustiveSearch" or "greedySearch".
            Defaults to "greedySearch".
        features_group (Optional[List[List[int]]], optional): A temporal matrix where each sublist contains indices of a
            longitudinal attribute's waves. Required for the longitudinal component. Defaults to None.
        parallel (bool, optional): Whether to use parallel processing for CFS (useful for exhaustive search with multiple
            groups). Defaults to False.
        outer_search_method (str, optional): The search method for Phase 2 (outer search). If None, defaults to
            `search_method`. Defaults to None.
        inner_search_method (str, optional): The search method for Phase 1 (inner search). Defaults to "exhaustiveSearch".
        version (int, optional): The version of the CFS-Per-Group algorithm to use. Version 1 applies CFS per group
            without an outer search, while Version 2 includes an outer CFS on the aggregated features. Defaults to 1.
        num_cpus (int, optional): Number of CPUs for parallel processing. If -1, uses all available CPUs. Defaults to -1.

    Attributes:
        selected_features_ (ndarray): Indices of the selected features after fitting.

    Examples:
        Below are examples demonstrating the usage of the `CorrelationBasedFeatureSelectionPerGroup` class.

        !!! example "Basic Usage with Longitudinal Component"
            ```python
            from scikit_longitudinal.preprocessors.feature_selection.correlation_feature_selection import CorrelationBasedFeatureSelectionPerGroup
            from scikit_longitudinal.data_preparation import LongitudinalDataset
            from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import LongitudinalEnsemblingStrategy


            # Load dataset
            dataset = LongitudinalDataset('./stroke_longitudinal.csv')
            dataset.load_data()
            dataset.load_target(target_column="stroke_w2")
            dataset.setup_features_group("elsa")
            dataset.load_train_test_split(test_size=0.2, random_state=42)

            # Initialize CFS-Per-Group
            cfs_longitudinal = CorrelationBasedFeatureSelectionPerGroup(
                features_group=dataset.feature_groups(),
                non_longitudinal_features=dataset.non_longitudinal_features()
            )

            # Fit to data
            cfs_longitudinal.fit(dataset.X_train, dataset.y_train)

            # Transform data
            X_selected = cfs_longitudinal.apply_selected_features_and_rename(dataset.X_train, cfs_longitudinal.selected_features_)
            print(X_selected)
            ```

        !!! example "Using Parallel Processing"
            ```python
            # ... Same as above, but with parallel processing enabled ...

            # Initialize with parallel processing
            cfs_longitudinal = CorrelationBasedFeatureSelectionPerGroup(
                features_group=features_group,
                search_method="exhaustiveSearch",
                parallel=True, # Enable parallel processing
                num_cpus=4 # Specify number of CPUs to use, -1 for all available CPUs
            )

            # ... Same as above, but with parallel processing enabled ...
            ```

        !!! example "Using Version 2 with Outer Search"
            ```python
            # ... Same as above, but with parallel processing enabled ...


            # Initialize with version 2 and outer search method
            cfs_longitudinal = CorrelationBasedFeatureSelectionPerGroup(
                features_group=features_group,
                non_longitudinal_features=non_longitudinal_features,
                version=2, # Use version 2 of CFS-Per-Group
                outer_search_method="greedySearch" # Specify outer search method
            )

            # ... Same as above, but with parallel processing enabled ...
            ```

    Notes:
        The CFS-Per-Group algorithm is based on the following references:

        - **Zixiao Shen's CFS Implementation**:
          - *Zixiao. S.* (2019, August 11). GitHub - ZixiaoShen/Correlation-based-Feature-Selection. Available at: [GitHub](https://github.com/ZixiaoShen/Correlation-based-Feature-Selection)
        - **Mastervii's CFS 2-Phase Variant**:
          - *Pomsuwan, T.* (2023, February 24). GitHub - mastervii/CSF_2-phase-variant. Available at: [GitHub](https://github.com/mastervii/CSF_2-phase-variant)
        - **Longitudinal Component References**:
          - **Version 1**:
            - *Pomsuwan, T. and Freitas, A.A.* (2017, November). Feature selection for the classification of longitudinal human ageing data. In *2017 IEEE International Conference on Data Mining Workshops (ICDMW)* (pp. 739-746). IEEE.
          - **Version 2**:
            - *Pomsuwan, T. and Freitas, A.A.* (2018, February). Feature selection for the classification of longitudinal human ageing data. Master's thesis, University of Kent. Available at: [University of Kent](https://kar.kent.ac.uk/66568/)

    See also:
        - `CorrelationBasedFeatureSelection`: For the standard CFS algorithm without the longitudinal component.
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
        """Fit the CFS-Per-Group algorithm to the data.

        This method applies the CFS-Per-Group algorithm, selecting features that are highly correlated with the target
        while minimizing redundancy within feature groups.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            y (np.ndarray): Target variable of shape (n_samples,).

        Returns:
            CorrelationBasedFeatureSelectionPerGroup: The fitted instance.
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
        """Transform the data by selecting the chosen features.

        This method is overridden from `CustomTransformerMixinEstimator` and selects the features based on
        `selected_features_`.

        !!! warning "Usage Note"
            Not to be used directly. Use the `apply_selected_features_and_rename` method instead.
            CFS Per Group has a specific behavior for longitudinal features, and this method does not
            account for that. It is recommended to use the `apply_selected_features_and_rename` method
            for proper handling of longitudinal features.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data with selected features.
        """
        return X

    def _fit_subset(self, X: np.ndarray, y: np.ndarray, group: Tuple[int]) -> List[int]:
        """Fit CFS on a specific feature group.

        This method applies the CFS algorithm to a subset of features defined by the group, selecting the most relevant
        features within that group.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Target variable.
            group (Tuple[int]): Indices of features in the group.

        Returns:
            List[int]: Selected feature indices from the group.
        """
        X_group = X[:, group]
        self._fit(X_group, y)
        return [group[i] for i in self.selected_features_]

    @ray.remote
    def _ray_fit_subset(self, X: np.ndarray, y: np.ndarray, group: Tuple[int]) -> List[int]:
        """Ray remote function for parallel fitting of CFS on a feature group.

        This method enables parallel processing of feature groups using Ray, which is particularly useful for large
        datasets or computationally intensive search methods.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Target variable.
            group (Tuple[int]): Indices of features in the group.

        Returns:
            List[int]: Selected feature indices from the group.
        """
        return self._fit_subset(X, y, group)

    # pylint: disable=W9016
    @staticmethod
    def apply_selected_features_and_rename(
        df: pd.DataFrame, selected_features: List, regex_match=r"^(.+)_w(\d+)$"
    ) -> [pd.DataFrame, None]:
        """Apply selected features to the DataFrame and rename non-longitudinal features.

        This method selects the specified features from the DataFrame and renames any features that, after selection,
        appear as single-wave features (i.e., non-longitudinal). This ensures that such features are not misinterpreted
        as longitudinal in future processing.

        !!! warning "Usage Note"
            This method should be used instead of the standard `transform` method to handle both feature selection and
            renaming in one step, especially in pipelines where the temporal structure needs to be preserved.

        !!! question "Regex Match, what is that all about?"
            The regex match is used to identify features that are longitudinal in nature. The default pattern
            `^(.+)_w(\d+)$` captures features with a base name followed by a wave number (e.g., `feature_w1`, `feature_w2`).
            Working by default with the ELSA databases in a nutshell.

            The first group `(.+)` captures the base name of the feature, while the second group `(\d+)` captures the wave
            number. This allows the method to identify and rename features that are longitudinal in nature, ensuring that
            they are treated correctly in subsequent analyses.

            Why is that important? Because we want to make sure that the features are not misinterpreted as longitudinal
            when they are actually single-wave features. This is particularly important in longitudinal datasets where
            features are collected over multiple time points.

        Args:
            df (pd.DataFrame): Input DataFrame.
            selected_features (List): List of selected feature indices.
            regex_match (str, optional): Regex pattern to identify wave-based features. Defaults to "^(.+)_w(\d+)$".

        Returns:
            pd.DataFrame: DataFrame with selected features and renamed non-longitudinal features.
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
