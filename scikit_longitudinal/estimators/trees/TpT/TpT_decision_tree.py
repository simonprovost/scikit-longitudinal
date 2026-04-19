# pylint: disable=too-many-arguments,invalid-name,signature-differs,no-member,R0801

import math
import warnings
from numbers import Real
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.utils._param_validation import Interval

from ._preprocessing import long_to_wide


class TpTDecisionTreeClassifier(DecisionTreeClassifier):
    """
    Time-penalised Trees (TpT) Decision Tree Classifier for longitudinal data classification.

    This classifier extends the standard Decision Tree algorithm to handle longitudinal data by incorporating a
    **time-penalised split gain**. At a parent node time $t_p$, a candidate split at time $t_c$ has gain
    $\\Delta I$ which is penalised as $\\Delta I \\cdot e^{-\\gamma (t_c - t_p)}$. In this Phase-1
    implementation, $t_c$ is proxied by the **wave index** of the splitting feature; in a later step we will
    propagate the true parent time through the builder to compute $t_c - t_p$ exactly.

    ??? note "LONG vs wide input — *[Soon To Be Deprecated](https://github.com/simonprovost/scikit-longitudinal/issues/64)*"
        TpT internally operates on a **wide** matrix (features expanded over waves). If `assume_long_format=True`,
        the classifier can accept a LONG-format dataframe and will convert it to the expected wide representation
        before fitting (using `id_col`, `time_col`, `duration_col`, `time_step`, and `max_horizon`).

        - LONG-format: one row per (subject, time) observation.
        - wide format: one row per subject, with features duplicated across waves.

        The conversion fills feature values up to each subject's duration/horizon and leaves NaNs beyond,
        enabling "duration leaves" in the TpT logic.

    Args:
        gamma (float, optional):
            Time-penalty rate $\\gamma$ in the factor $e^{-\\gamma \\Delta t}$.
            If not provided, falls back to `threshold_gain` (for backward compatibility).
        threshold_gain (float, optional):
            Backward-compatible alias for `gamma`. If both are provided, `gamma` takes precedence.
        features_group (List[List[int]], optional):
            A list of lists where each inner list contains indices of features corresponding to a specific longitudinal
            attribute across different waves. The order within each inner list reflects the temporal sequence, with the
            first element being the oldest wave and the last being the most recent. For example, `[[0,1],[2,3]]` indicates
            two longitudinal attributes, each with two waves (e.g., 0: oldest, 1: recent; 2: oldest, 3: recent).
        max_horizon (int, optional):
            Maximum temporal horizon for predictions. Limits the time range explored during tree construction.
            If None, no limit is applied.
        criterion (str, default="entropy"):
            The function to measure the quality of a split. Fixed to "entropy" for this algorithm; do not change.
        splitter (str, default="TpT"):
            The strategy used to choose the split at each node. Fixed to "TpT" for this algorithm; do not change.
        max_depth (Optional[int], default=None):
            The maximum depth of the tree. If None, nodes are expanded until all leaves are pure or meet other constraints.
        min_samples_split (int, default=2):
            The minimum number of samples required to split an internal node.
        min_samples_leaf (int, default=1):
            The minimum number of samples required to be at a leaf node.
        min_weight_fraction_leaf (float, default=0.0):
            The minimum weighted fraction of the sum total of weights required to be at a leaf node.
        max_features (Optional[Union[int, str]], default=None):
            The number of features to consider when looking for the best split. Can be int, float, "auto", "sqrt", or "log2".
        random_state (Optional[int], default=None):
            The seed used by the random number generator for reproducibility.
        max_leaf_nodes (Optional[int], default=None):
            The maximum number of leaf nodes in the tree. If None, unlimited.
        min_impurity_decrease (float, default=0.0):
            The minimum impurity decrease required for a node to be split.
        class_weight (Optional[Union[dict, str]], default=None):
            Weights associated with classes in the form `{class_label: weight}` or "balanced".
        ccp_alpha (float, default=0.0):
            Complexity parameter used for Minimal Cost-Complexity Pruning. Must be non-negative.
        store_leaf_values (bool, default=False):
            Whether to store leaf values during tree construction.
        monotonic_cst (Optional[List[int]], default=None):
            Monotonic constraints for features (1 for increasing, -1 for decreasing, 0 for no constraint).
        min_penalized_gain (float, default=0.0):
            Minimum normalized time-penalized gain required to keep a split. Mirrors `min_criterion` from the reference
            TpT Python implementation; for the PBC study, set `min_penalized_gain=5e-7` to reproduce the depth-4 tree.

    Attributes:
        classes_ (ndarray of shape (n_classes,)):
            The class labels.
        n_classes_ (int):
            The number of classes.
        n_features_ (int):
            The number of features when fit is performed.
        n_outputs_ (int):
            The number of outputs when fit is performed (fixed to 1 for this classifier).
        feature_importances_ (ndarray of shape (n_features,)):
            The impurity-based feature importances.
        max_features_ (int):
            The inferred value of max_features after fitting.
        tree_ (Tree object):
            The underlying Tree object representing the decision tree.

    Examples:
        Below are examples demonstrating the usage of the `TpTDecisionTreeClassifier` class.

        !!! example "Basic Usage"

            Please note that the Iris is not longitudinal data, but this example is for demonstration purposes only.
            We could not publicly use the dataset we use for our various papers without user registering
            to the [ELSA](https://www.elsa-project.ac.uk/) project.

            If you find public longitudinal datasets, or if you have also more public-yet-registration-required
            datasets / private datasets, please adapt the examples to your usecase.

            ```python
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            from scikit_longitudinal.estimators.trees import TpTDecisionTreeClassifier

            # Load dataset
            X, y = load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Define features_group (example for illustration; adjust based on actual longitudinal structure)
            features_group = [[0, 1], [2, 3]]

            # Initialize and fit the classifier
            clf = TpTDecisionTreeClassifier(gamma=0.1, features_group=features_group)
            clf.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy}")
            ```

        !!! example "Advanced: using with LongitudinalPipeline"

            ```python
            from scikit_longitudinal.pipeline import LongitudinalPipeline
            from scikit_longitudinal.data_preparation import LongitudinalDataset
            from scikit_longitudinal.estimators.trees import TpTDecisionTreeClassifier
            from scikit_longitudinal.data_preparation import LongitudinalDataset
            from scikit_longitudinal.data_preparation import MerWavTimePlus

            # Load dataset
            dataset = LongitudinalDataset('./stroke_longitudinal.csv')
            dataset.load_data()
            dataset.load_target(target_column="stroke_w2")
            dataset.setup_features_group("elsa")
            dataset.load_train_test_split(test_size=0.2, random_state=42)

            # Define pipeline steps with TpTDecisionTreeClassifier
            steps = [
                ('MerWavTime Plus', MerWavTimePlus()), # Recall, a pipeline is at least two steps and the first one being a Data Transformation step. Here as we use a Longitudinal classifier, we need to use MerWavTimePlus, retaining the temporal dependency.
                ('classifier', TpTDecisionTreeClassifier(features_group=dataset.feature_groups()))
            ]

            # Initialize pipeline
            pipeline = LongitudinalPipeline(
                steps=steps,
                features_group=dataset.feature_groups(),
                non_longitudinal_features=dataset.non_longitudinal_features(),
                feature_list_names=dataset.data.columns.tolist(),
                update_feature_groups_callback="default"
            )

            # Fit and predict
            pipeline.fit(dataset.X_train, dataset.y_train)
            y_pred = pipeline.predict(dataset.X_test)
            print(f"Predictions: {y_pred}")
            ```
    """

    _parameter_constraints = {
        **DecisionTreeClassifier._parameter_constraints,
        "gamma": [Interval(Real, 0.0, None, closed="left")],
        "threshold_gain": [Interval(Real, 0.0, None, closed="left")],
    }

    def __init__(
        self,
        gamma: Optional[float] = None,
        threshold_gain: Optional[float] = None,  # deprecated alias for gamma
        features_group: Optional[List[List[int]]] = None,
        max_horizon: Optional[int] = None,
        id_col: Optional[str] = None,
        time_col: Optional[str] = None,
        duration_col: Optional[str] = None,
        time_step: float = 1.0,
        assume_long_format: bool = False,
        long_feature_columns: Optional[List[str]] = None,
        criterion: str = "entropy",
        splitter: str = "TpT",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[Union[int, str]] = None,
        random_state: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        class_weight: Optional[str] = None,
        ccp_alpha: float = 0.0,
        store_leaf_values: bool = False,
        monotonic_cst: Optional[List[int]] = None,
        min_penalized_gain: float = 0.0,
    ):
        # Resolve gamma with backward-compatible alias
        _gamma = gamma if gamma is not None else (threshold_gain if threshold_gain is not None else 0.0015)
        self.gamma = float(_gamma)
        # Keep attribute name threshold_gain for downstream Cython param compatibility
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
        self.min_penalized_gain = float(min_penalized_gain)

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
            threshold_gain=self.threshold_gain,
            features_group=self.features_group,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            store_leaf_values=store_leaf_values,
            monotonic_cst=monotonic_cst,
        )

    def fit(self, X, y, *args, **kwargs):
        """Fit the classifier, optionally preparing LONG-format data on the fly."""

        X_prepared, y_prepared = self._prepare_training_data(X, y)

        fitted = super().fit(X_prepared, y_prepared, *args, **kwargs)

        if self.min_penalized_gain > 0.0 and hasattr(self, "tree_"):
            self._prune_penalized_gain()

        return fitted

    # ------------------------------------------------------------------
    # Data preparation utilities
    # ------------------------------------------------------------------
    def _ensure_series(self, y, X) -> pd.Series:
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("Target dataframe must have exactly one column for classification.")
            y_series = y.iloc[:, 0]
        elif isinstance(y, pd.Series):
            y_series = y
        else:
            y_series = pd.Series(np.asarray(y).ravel(), index=getattr(X, "index", None))

        if getattr(X, "shape", None) is not None and len(y_series) != len(X):
            raise ValueError("Target vector length must match number of observations.")

        if y_series.isna().any():
            raise ValueError("NaN targets are not supported in TpTDecisionTreeClassifier.")

        return y_series

    def _prepare_training_data(self, X, y):
        """Return (X_prepared, y_prepared), handling LONG-format input if required."""

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

        wide_data.y = reindexed_targets.to_numpy()

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

        expected_features = getattr(self, "_expected_n_features_", None)
        if expected_features is None:
            expected_features = data.X.shape[1]
        if data.X.shape[1] < expected_features:
            pad = np.full((data.X.shape[0], expected_features - data.X.shape[1]), np.nan)
            X_wide = np.concatenate([data.X, pad], axis=1)
        elif data.X.shape[1] > expected_features:
            X_wide = data.X[:, :expected_features]
        else:
            X_wide = data.X

        return X_wide, data.subject_ids

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def predict(self, X):  # type: ignore[override]
        if self._uses_long_format and isinstance(X, pd.DataFrame):
            X_wide, _ = self._prepare_long_dataset(X)
            return super().predict(X_wide)
        return super().predict(X)

    def predict_proba(self, X):  # type: ignore[override]
        if self._uses_long_format and isinstance(X, pd.DataFrame):
            X_wide, _ = self._prepare_long_dataset(X)
            return super().predict_proba(X_wide)
        return super().predict_proba(X)

    def _more_tags(self):
        tags = super()._more_tags()
        tags["allow_nan"] = True
        return tags

    # --------------------------------------------------------------------- #
    # Internal utilities
    # --------------------------------------------------------------------- #
    def _prune_penalized_gain(self) -> None:
        """Post-prune leaves whose penalized gain falls below the notebook threshold."""
        tree_ = self.tree_
        children_left = tree_.children_left
        children_right = tree_.children_right
        features = tree_.feature
        thresholds = tree_.threshold
        impurities = tree_.impurity
        split_times = tree_.split_time_index
        impurity_duration = getattr(tree_, "impurity_duration", None)
        n_node_samples = tree_.n_node_samples

        total_samples = float(n_node_samples[0]) if n_node_samples.size else 0.0
        if total_samples <= 0.0:
            return

        tree_leaf = _tree.TREE_LEAF
        tree_undefined = _tree.TREE_UNDEFINED
        min_gain = self.min_penalized_gain
        gamma = self.threshold_gain

        def prune_node(node_id: int, parent_time_index: float) -> None:
            left = children_left[node_id]
            right = children_right[node_id]

            if left != tree_leaf:
                prune_node(left, split_times[node_id])
                left = children_left[node_id]
            if right != tree_leaf:
                prune_node(right, split_times[node_id])
                right = children_right[node_id]

            if left == tree_leaf and right == tree_leaf:
                gain_ratio = compute_gain_ratio(node_id, parent_time_index)
                if gain_ratio < min_gain:
                    children_left[node_id] = tree_leaf
                    children_right[node_id] = tree_leaf
                    features[node_id] = tree_undefined
                    thresholds[node_id] = tree_undefined
                    split_times[node_id] = parent_time_index
                    if impurity_duration is not None:
                        tree_.impurity_duration[node_id] = np.inf

        def compute_gain_ratio(node_id: int, parent_time_index: float) -> float:
            n_node = float(n_node_samples[node_id])
            if n_node <= 0.0:
                return 0.0

            left = children_left[node_id]
            right = children_right[node_id]

            left_count = float(n_node_samples[left]) if left != tree_leaf else 0.0
            right_count = float(n_node_samples[right]) if right != tree_leaf else 0.0
            duration_count = max(0.0, n_node - left_count - right_count)

            weighted_impurity = 0.0
            if left_count > 0.0:
                weighted_impurity += (left_count / n_node) * impurities[left]
            if right_count > 0.0:
                weighted_impurity += (right_count / n_node) * impurities[right]
            if duration_count > 0.0 and impurity_duration is not None:
                duration_impurity = tree_.impurity_duration[node_id]
                if np.isfinite(duration_impurity):
                    weighted_impurity += (duration_count / n_node) * duration_impurity

            unpenalized_gain = impurities[node_id] - weighted_impurity
            delta_t = max(0.0, float(split_times[node_id]) - float(parent_time_index))
            penalized_gain = unpenalized_gain * math.exp(-gamma * delta_t)

            return penalized_gain * (n_node / total_samples)

        prune_node(0, 0.0)

        depths = tree_.compute_node_depths()
        tree_.max_depth = int(np.max(depths)) if depths.size else 0
