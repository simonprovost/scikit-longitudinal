from __future__ import annotations

import warnings
from numbers import Real
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils._param_validation import Interval

from ._preprocessing import long_to_wide


class TpTDecisionTreeRegressor(DecisionTreeRegressor):
    """
    Time-penalised Trees (TpT) Decision Tree Regressor for longitudinal data regression.

    This regressor extends scikit-learn's `DecisionTreeRegressor` for longitudinal data by incorporating a
    **time-penalised split gain**. At a node associated with a parent time $t_p$, a candidate split evaluated
    at time $t_c$ yields an impurity improvement $\\Delta I$ (typically based on variance reduction / MSE),
    which is penalised as $G_\\gamma = \\Delta I \\cdot e^{-\\gamma (t_c - t_p)}$. In the current implementation,
    $t_c$ is represented by the **wave index** of the splitting feature and $t_p$ is propagated by the tree
    builder, so that the penalty depends on the *time distance* between successive splits.

    ??? note "LONG vs wide input — *[Soon To Be Deprecated](https://github.com/simonprovost/scikit-longitudinal/issues/64)*"
        TpT internally operates on a **wide** matrix (features expanded over waves). If `assume_long_format=True`,
        the regressor can accept a LONG-format dataframe and will convert it to the expected wide representation
        before fitting (using `id_col`, `time_col`, `duration_col`, `time_step`, and `max_horizon`).

        - LONG-format: one row per (subject, time) observation.
        - wide format: one row per subject, with features duplicated across waves.

        The conversion fills feature values up to each subject's duration/horizon and leaves NaNs beyond,
        enabling "duration leaves" in the TpT logic.

    Args:
        gamma (float, optional):
            Time-penalty rate $\\gamma$ in $e^{-\\gamma \\Delta t}$. If not provided, falls back to
            `threshold_gain` for backward compatibility.
        threshold_gain (float, optional):
            Backward-compatible alias for `gamma`. If both are provided, `gamma` takes precedence. (Internally
            reused to match existing Cython parameter naming.)
        features_group (List[List[int]], optional):
            Temporal grouping of feature indices (waves per covariate). Required when using wide-format input.
            If `assume_long_format=True`, this can be inferred/constructed during preprocessing depending on
            how wide features are generated.
        max_horizon (int, optional):
            Optional cap for the horizon considered during LONG-format preprocessing.
        id_col (str, optional):
            Subject identifier column name in LONG-format.
        time_col (str, optional):
            Observation time column name in LONG-format.
        duration_col (str, optional):
            Subject-specific horizon/duration column name in LONG-format.
        time_step (float, default=1.0):
            Temporal discretisation step used to map times to wave indices.
        assume_long_format (bool, default=False):
            If True, interpret `X` as LONG-format and convert to wide prior to fitting.
        long_feature_columns (List[str], optional):
            Subset of LONG-format columns to treat as features.
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
        _wide_feature_names_ (List[str]):
            Names of generated wide features (when LONG-format preprocessing is enabled).
        _subject_ids_ (List[str]):
            Subject ids aligned with the wide matrix rows (when LONG-format preprocessing is enabled).

    Examples:
        !!! example "Basic Usage"
            ```python
            import pandas as pd
            from scikit_longitudinal.estimators.trees import TpTDecisionTreeRegressor

            df_long = pd.read_csv("my_longitudinal_dataset.csv")
            y = df_long["target"]
            X = df_long.drop(columns=["target"])

            reg = TpTDecisionTreeRegressor(
                gamma=0.01,
                assume_long_format=True,
                id_col="id",
                time_col="time_point",
                duration_col="duration",
                time_step=1.0,
                max_depth=4,
                random_state=0,
            )
            reg.fit(X, y)
            preds = reg.predict(X)
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
        max_horizon: Optional[int] = None,
        id_col: Optional[str] = None,
        time_col: Optional[str] = None,
        duration_col: Optional[str] = None,
        time_step: float = 1.0,
        assume_long_format: bool = False,
        long_feature_columns: Optional[List[str]] = None,
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
        self.max_horizon = max_horizon
        self.id_col = id_col
        self.time_col = time_col
        self.duration_col = duration_col
        self.time_step = float(time_step)
        self.assume_long_format = assume_long_format
        self.long_feature_columns = long_feature_columns
        self._uses_long_format = False
        self._expected_n_features_: Optional[int] = None

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


    def _ensure_series(self, y, X) -> pd.Series:
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("Target dataframe must have exactly one column for regression.")
            y_series = y.iloc[:, 0]
        elif isinstance(y, pd.Series):
            y_series = y
        else:
            y_series = pd.Series(np.asarray(y).ravel(), index=getattr(X, "index", None))

        if getattr(X, "shape", None) is not None and len(y_series) != len(X):
            raise ValueError("Target vector length must match number of observations.")
        if y_series.isna().any():
            raise ValueError("NaN targets are not supported in TpTDecisionTreeRegressor.")
        return y_series

    def _prepare_training_data(self, X, y):
        long_format = self.assume_long_format or (
            self.features_group is None and isinstance(X, pd.DataFrame)
        )

        if not long_format:
            self._uses_long_format = False
            if self.features_group is None:
                raise ValueError(
                    "When providing wide-format data you must also set features_group to describe waves."
                )
            return X, y

        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "LONG-format input requires a pandas DataFrame with id/time/duration columns."
            )

        if not all([self.id_col, self.time_col, self.duration_col]):
            raise ValueError(
                "id_col, time_col and duration_col must be specified to consume LONG-format data."
            )

        y_series = self._ensure_series(y, X)
        wide_data = long_to_wide(
            X,
            id_col=self.id_col,
            time_col=self.time_col,
            duration_col=self.duration_col,
            time_step=self.time_step,
            max_horizon=self.max_horizon,
            feature_columns=self.long_feature_columns,
        )

        subject_targets = (
            y_series.groupby(X[self.id_col]).first().rename(index=lambda idx: str(idx))
        )
        reindexed_targets = subject_targets.reindex(wide_data.subject_ids)
        if reindexed_targets.isna().any():
            missing = reindexed_targets[reindexed_targets.isna()].index.tolist()
            raise ValueError(
                "Missing target values for subjects after aggregation: " + ", ".join(map(str, missing))
            )

        wide_data.y = reindexed_targets.to_numpy(dtype=np.float64)

        self.features_group = wide_data.feature_groups
        self._uses_long_format = True
        self._wide_feature_names_ = wide_data.feature_names
        self._subject_ids_ = wide_data.subject_ids
        self._feature_columns_long_ = list(wide_data.feature_columns)
        self._n_waves_ = len(wide_data.time_indices)
        self._expected_n_features_ = wide_data.X.shape[1]

        return wide_data.X, wide_data.y

    def _prepare_long_dataset(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        if not self._uses_long_format:
            raise ValueError("This estimator was not fitted on LONG-format data.")

        data = long_to_wide(
            X,
            id_col=self.id_col,
            time_col=self.time_col,
            duration_col=self.duration_col,
            time_step=self.time_step,
            max_horizon=self.max_horizon,
            feature_columns=self._feature_columns_long_,
        )

        expected = getattr(self, "_expected_n_features_", None)
        if expected is None:
            expected = data.X.shape[1]

        if data.X.shape[1] < expected:
            pad = np.full((data.X.shape[0], expected - data.X.shape[1]), np.nan)
            X_wide = np.concatenate([data.X, pad], axis=1)
        elif data.X.shape[1] > expected:
            X_wide = data.X[:, :expected]
        else:
            X_wide = data.X

        return X_wide, data.subject_ids


    def fit(self, X, y, *args, **kwargs):  # type: ignore[override]
        X_prepared, y_prepared = self._prepare_training_data(X, y)
        return super().fit(X_prepared, y_prepared, *args, **kwargs)

    def predict(self, X):  # type: ignore[override]
        if self._uses_long_format and isinstance(X, pd.DataFrame):
            X_wide, _ = self._prepare_long_dataset(X)
            return super().predict(X_wide)
        return super().predict(X)

    def _more_tags(self):
        tags = super()._more_tags()
        tags["allow_nan"] = True
        return tags
