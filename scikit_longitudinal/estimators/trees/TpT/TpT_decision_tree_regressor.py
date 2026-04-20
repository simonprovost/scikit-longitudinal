from __future__ import annotations

import warnings
from numbers import Real
from typing import List, Optional, Union

from sklearn.tree import DecisionTreeRegressor
from sklearn.utils._param_validation import Interval


class TpTDecisionTreeRegressor(DecisionTreeRegressor):
    """
    Time-penalised Trees (TpT) Decision Tree Regressor for longitudinal data regression.

    This regressor extends scikit-learn's `DecisionTreeRegressor` for longitudinal data by incorporating a
    **time-penalised split gain**. At a node associated with a parent time $t_p$, a candidate split evaluated
    at time $t_c$ yields an impurity improvement $\\Delta I$ (typically based on variance reduction / MSE),
    which is penalised as $G_\\gamma = \\Delta I \\cdot e^{-\\gamma (t_c - t_p)}$. In the current implementation,
    $t_c$ is encoded in the **wave index** of the splitting feature and $t_p$ is propagated by the tree
    builder, so the penalty depends on the *time distance* between successive splits. The splitter therefore
    tends to prefer earlier waves (while allowing later waves deeper in the tree) unless later observations
    bring a substantially stronger signal.

    Args:
        gamma (float, optional):
            Time-penalty rate $\\gamma$ in $e^{-\\gamma \\Delta t}$. If not provided, falls back to
            `threshold_gain`.
        threshold_gain (float, optional):
            Alias for `gamma`. If both are provided, `gamma` takes precedence. (Internally
            reused to match existing Cython parameter naming.)
        features_group (List[List[int]], optional):
            Temporal grouping of feature indices (waves per covariate).
        criterion (str, default="friedman_mse"):
            Split criterion for regression. The intended criterion is MSE / variance reduction. (Other criteria
            may not be supported depending on the current Cython implementation.)
        splitter (str, default="TpT"):
            Split strategy identifier. Must match the TpT splitter name exposed by the underlying Cython backend.
        max_depth (Optional[int], default=None):
            Maximum depth of the tree. If None, the tree expands until other stopping criteria apply.
        min_samples_split (int, default=2):
            Minimum number of samples required to split an internal node.
        min_samples_leaf (int, default=1):
            Minimum number of samples required to be at a leaf node.
        min_weight_fraction_leaf (float, default=0.0):
            Minimum weighted fraction of the sum of weights required in each leaf.
        max_features (Optional[Union[int, float, str]], default=None):
            Number of features to consider at each split.
        random_state (Optional[int], default=None):
            Controls randomness of feature sampling and tie-breaking.
        max_leaf_nodes (Optional[int], default=None):
            Grow a tree with at most `max_leaf_nodes` leaves (best-first strategy when supported).
        min_impurity_decrease (float, default=0.0):
            Minimum (unpenalised) impurity decrease required to split.
        ccp_alpha (float, default=0.0):
            Complexity parameter used for Minimal Cost-Complexity Pruning.
        store_leaf_values (bool, default=False):
            Whether to store the samples that fall into leaves in the `tree_` attribute.
        monotonic_cst (Optional[List[int]], default=None):
            Monotonic constraints for features (if supported by the underlying sklearn tree code and compatible
            with missing values / regression settings).

    Attributes:
        n_features_in_ (int):
            Number of features seen during fit (wide representation).
        tree_ (sklearn.tree._tree.Tree):
            The underlying fitted tree structure.
        feature_importances_ (ndarray of shape (n_features,)):
            Impurity-based feature importances (variance reduction based).

    Examples:
        !!! example "Basic Usage"
            ```python
            from scikit_longitudinal.estimators.trees import TpTDecisionTreeRegressor

            features_group = [[0, 1], [2, 3]]
            reg = TpTDecisionTreeRegressor(
                gamma=0.01,
                features_group=features_group,
                max_depth=4,
                random_state=0,
            )
            reg.fit(X_wide, y)
            preds = reg.predict(X_wide)
            ```
    """

    _parameter_constraints = {
        **DecisionTreeRegressor._parameter_constraints,
        "gamma": [Interval(Real, 0.0, None, closed="left")],
        "threshold_gain": [Interval(Real, 0.0, None, closed="left")],
    }

    def __init__(
        self,
        gamma: Optional[float] = None,
        threshold_gain: Optional[float] = None,
        features_group: Optional[List[List[int]]] = None,
        criterion: str = "friedman_mse",
        splitter: str = "TpT",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        store_leaf_values: bool = False,
        monotonic_cst: Optional[List[int]] = None,
    ) -> None:
        _gamma = gamma if gamma is not None else (threshold_gain if threshold_gain is not None else 0.0015)
        self.gamma = float(_gamma)
        self.threshold_gain = self.gamma
        self.features_group = features_group

        if monotonic_cst is not None:
            warnings.warn(
                "TpT does not currently support monotonic constraints; "
                "monotonic_cst is being forced to None.",
                UserWarning,
                stacklevel=2,
            )
            monotonic_cst = None

        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            store_leaf_values=store_leaf_values,
            monotonic_cst=monotonic_cst,
            threshold_gain=self.threshold_gain,
            features_group=self.features_group,
        )

    def fit(self, X, y, *args, **kwargs):  # type: ignore[override]
        """
        Fit the Time-penalised Trees (TpT) Decision Tree Regressor to the training data.

        This method trains the regressor using the provided training data and targets. It requires the `features_group`
        parameter to be set, as the time-penalised splitter relies on it to read the wave index of each candidate split.

        Args:
            X (array-like of shape (n_samples, n_features)):
                The training input samples in wide format (features expanded over waves).
            y (array-like of shape (n_samples,)):
                The target values (continuous).
            *args:
                Additional positional arguments passed to the superclass `fit` method.
            **kwargs:
                Additional keyword arguments passed to the superclass `fit` method.

        Returns:
            TpTDecisionTreeRegressor:
                The fitted regressor instance.

        Raises:
            ValueError:
                If `features_group` is not provided, as it is required for longitudinal functionality.
        """
        if self.features_group is None:
            raise ValueError("The features_group parameter must be provided.")
        return super().fit(X, y, *args, **kwargs)

    def _more_tags(self):
        tags = super()._more_tags()
        tags["allow_nan"] = True
        return tags
