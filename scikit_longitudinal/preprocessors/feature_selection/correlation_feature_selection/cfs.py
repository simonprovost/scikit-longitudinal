import numpy as np
import ray
from overrides import override

from scikit_longitudinal.preprocessors.feature_selection.correlation_feature_selection.utils import (
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
        parrallel : bool, default=False
            A boolean that indicates whether to use parallel computation. It means that the search will be performed
            using Ray, for each candidate feature subset it will calculate the merit in parallel. Further details
            are provided in the search algorithms themselves, in the utils.py file.
        num_cpus : int, default=0
            The number of CPUs to use for parallel computation. Used only if parrallel is True. If 0, it has no effect.

    Attributes:
        selected_features_ : ndarray of shape (n_features,)
            The indices of the selected features.

    Examples:
        >>>  # Without the longitudinal component (original CFS):
        >>> import numpy as np
        >>> from sklearn_fork.datasets import make_classification
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
            parallel: bool = False,
            num_cpus: int = -1,
    ):
        if search_method not in {
            "exhaustiveSearch",
            "greedySearch",
        }:
            raise ValueError("search_method must be: 'exhaustiveSearch', or 'greedySearch'")

        if not isinstance(parallel, bool):
            raise ValueError("parallel must be a boolean.")
        if not isinstance(num_cpus, int):
            raise ValueError("num_cpus must be an integer.")
        if parallel and num_cpus <= 0:
            raise ValueError("num_cpus must be greater than 0.")

        if parallel and not ray.is_initialized():
            ray.init(num_cpus=num_cpus)

        self._parallel = parallel
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
            self.selected_features_ = _exhaustive_search(X, y, self._parallel)
        elif self.search_method == "greedySearch":
            self.selected_features_ = _greedy_search(X, y, self._parallel)
        else:
            raise ValueError(f"Search method {self.search_method} is not supported.")
        return self

    @override
    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms the input data to the selected features.

        Args:
            X (np.ndarray):
                Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: The transformed data of shape (n_samples, n_selected_features).

        """
        return X[:, self.selected_features_]
