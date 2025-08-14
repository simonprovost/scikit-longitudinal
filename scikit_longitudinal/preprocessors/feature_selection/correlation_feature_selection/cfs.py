import re
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from overrides import override

from scikit_longitudinal.preprocessors.feature_selection.correlation_feature_selection.algorithms import (
    _exhaustive_search,
    _greedy_search,
)
from scikit_longitudinal.templates import CustomTransformerMixinEstimator


# pylint: disable=R0902, R0801
class CorrelationBasedFeatureSelection(CustomTransformerMixinEstimator):
    """Correlation-based Feature Selection (CFS).

    This class performs feature selection using the correlation-based feature selection (CFS) algorithm on given data.
    The CFS algorithm is a filter method that selects features based on their correlation with the target variable and
    their mutual correlation with each other. This implementation supports the following search methods:
    exhaustiveSearch, or greedySearch. This implementation concern the support for CFS only. For the CFS per group
    (longitudinal component), refer to the CorrelationBasedFeatureSelectionPerGroup class.

    Read more in the Notes below for implementation details.

    Args:
        search_method : str, default="greedySearch"
            The search method to use. Options are "exhaustiveSearch", and "greedySearch".

    Attributes:
        selected_features_ : ndarray of shape (n_features,)
            The indices of the selected features.

    Examples:
        >>>  # Without the longitudinal component (original CFS):
        >>> import numpy as np
        >>> from sklearn.datasets import make_classification
        >>> from scikit_longitudinal import CorrelationBasedFeatureSelection
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> cfs = CorrelationBasedFeatureSelection()
        >>> cfs.fit(X, y)
        >>> X_selected = cfs.transform(X)
        >>> X_selected.shape
        >>> # (100, N) ; N is the number of selected features

    Notes:
        The improved CFS algorithm is based on the following references:

        * Zixiao. S. (2019, August 11). GitHub - ZixiaoShen
        /Correlation-based-Feature-Selection, available at:
        https://github.com/ZixiaoShen/Correlation-based-Feature-Selection

    See also:
        * CustomTransformerMixinEstimator: Base class for all Transformer Mixin estimators in scikit-learn that we
        customed so that the original scikit-learn "check_x_y" is performed all the time.

    """

    # pylint: disable=too-many-arguments,invalid-name,signature-differs,no-member
    def __init__(
        self,
        search_method: str = "greedySearch",
    ):
        assert search_method in {
            "exhaustiveSearch",
            "greedySearch",
        }, "search_method must be: 'exhaustiveSearch', or 'greedySearch'"

        self.search_method = search_method
        self.selected_features_ = []

    @override
    def _fit(self, X: np.ndarray, y: np.ndarray) -> "CorrelationBasedFeatureSelection":
        """Fits the feature selection algorithm to the given data.

        Args:
            X (np.ndarray):
                Input data of shape (n_samples, n_features).
            y (np.ndarray):
                Target variable of shape (n_samples).

        Returns:
            CorrelationBasedFeatureSelection: The fitted feature selection algorithm.

        """
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
            CFS has a specific behavior for longitudinal features, and this method does not
            account for that. It is recommended to use the `apply_selected_features_and_rename` method
            for proper handling of longitudinal features.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data with selected features.
        """
        return X

    @staticmethod
    def apply_selected_features_and_rename(
        df: pd.DataFrame, selected_features: List, regex_match=r"^(.+)_w(\d+)$"
    ) -> pd.DataFrame:
        """
        Apply selected features to the DataFrame and rename non-longitudinal features.

        Args:
            df (pd.DataFrame): Input DataFrame.
            selected_features (List): List of selected feature indices.
            regex_match (str, optional): Regex pattern to identify wave-based features.

        Returns:
            pd.DataFrame: DataFrame with selected features and renamed non-longitudinal features.
        """
        if selected_features:
            df = df.iloc[:, selected_features].copy()

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
