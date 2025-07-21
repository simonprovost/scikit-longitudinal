# pylint: disable=too-many-arguments,invalid-name,signature-differs,no-member,R0801

from typing import List, Optional, Union

from sklearn.tree import DecisionTreeRegressor


class LexicoDecisionTreeRegressor(DecisionTreeRegressor):
    """
    Lexico Decision Tree Regressor for longitudinal data regression.

    The `LexicoDecisionTreeRegressor` is a specialized regression model designed for longitudinal data. It builds
    upon scikit-learn's `DecisionTreeRegressor` by integrating a lexicographic optimization strategy. This approach
    prioritizes recent data points (waves) during split selection, optimizing both statistical accuracy and temporal
    relevance—a powerful tool for modeling time-dependent phenomena like patient health trends or economic forecasts.

    !!! question "How Does Lexicographic Optimization Work?"
        This regressor adapts the traditional decision tree algorithm for longitudinal data by considering two objectives:

        1. **Primary**: Maximize the information gain ratio (using "friedman_mse" criterion).
        2. **Secondary**: Favor features from more recent waves when gain ratios are comparable (within `threshold_gain`).

        This dual approach ensures that the tree leverages both statistical purity and temporal relevance.


    !!! note "Performance Boost with Cython"
        The underlying splitter (`node_lexicoRF_split`) is optimized in Cython for faster computation. Check out the
        [Cython implementation](https://github.com/simonprovost/scikit-lexicographical-trees/blob/21443b9dce51434b3198ccabac8bafc4698ce953/sklearn/tree/_splitter.pyx#L695)
        for a deep dive into the performance enhancements.

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
        threshold_gain (float, default=0.0015):
            Threshold for comparing gain ratios during split selection. Lower values prioritize recency more strictly;
            higher values allow more flexibility in balancing gain and recency.
        features_group (List[List[int]], optional):
            A list of lists where each sublist contains feature indices for a longitudinal attribute, ordered from
            oldest to most recent wave. Required for longitudinal functionality.
        criterion (str, default="friedman_mse"):
            The split quality metric. Fixed to "friedman_mse"; do not modify.
        splitter (str, default="lexicoRF"):
            The split strategy. Fixed to "lexicoRF"; do not modify.
        max_depth (Optional[int], default=None):
            Maximum tree depth. If None, grows until purity or other limits are reached.
        min_samples_split (int, default=2):
            Minimum samples required to split a node.
        min_samples_leaf (int, default=1):
            Minimum samples required at a leaf node.
        min_weight_fraction_leaf (float, default=0.0):
            Minimum weighted fraction of total sample weight at a leaf.
        max_features (Optional[Union[int, str]], default=None):
            Number of features to consider for splits (e.g., "auto", "sqrt", int).
        random_state (Optional[int], default=None):
            Seed for random number generation.
        max_leaf_nodes (Optional[int], default=None):
            Maximum number of leaf nodes.
        min_impurity_decrease (float, default=0.0):
            Minimum impurity decrease required for a split.
        ccp_alpha (float, default=0.0):
            Complexity parameter for pruning; non-negative.

    Attributes:
        n_features_ (int):
            Number of features in the fitted model.
        n_outputs_ (int):
            Number of outputs (fixed to 1 for regression).
        feature_importances_ (ndarray of shape (n_features,)):
            Impurity-based feature importances.
        max_features_ (int):
            Inferred value of `max_features` after fitting.
        tree_ (Tree object):
            The underlying decision tree structure.

    Examples:
        While `Sklong` focussed classification tasks only as of now. This regressor model is used by
        our LexicographicalGradientBoosting primitive. Feel free to experiment with it in your own
        longitudinal regression tasks but we do not guarantee its performance.

    Notes:
        - **Performance**: Best suited for longitudinal data; may not outperform standard regressors on non-temporal data.
        - **Reference**: Ribeiro, C. and Freitas, A., 2020. "A new random forest method for longitudinal data regression
          using a lexicographic bi-objective approach." In *2020 IEEE Symposium Series on Computational Intelligence
          (SSCI)* (pp. 806-813).
    """

    def __init__(
        self,
        threshold_gain: float = 0.0015,
        features_group: List[List[int]] = None,
        criterion: str = "friedman_mse",  # Do not change this value
        splitter: str = "lexicoRF",  # Do not change this value
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[Union[int, str]] = None,
        random_state: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        store_leaf_values: bool = False,
        monotonic_cst: Optional[List[int]] = None,
    ):
        self.threshold_gain = threshold_gain
        self.features_group = features_group

        super().__init__(
            criterion=criterion,
            threshold_gain=threshold_gain,
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
            ccp_alpha=ccp_alpha,
            store_leaf_values=store_leaf_values,
            monotonic_cst=monotonic_cst,
        )

    def fit(self, X, y, *args, **kwargs):
        """
        Fit the Lexico Decision Tree Regressor to the data.

        Args:
            X (array-like of shape (n_samples, n_features)): Training input samples.
            y (array-like of shape (n_samples,)): Target values.
            *args: Additional positional arguments for the superclass `fit`.
            **kwargs: Additional keyword arguments for the superclass `fit`.

        Returns:
            self: Fitted regressor instance.

        Raises:
            ValueError: If `features_group` is not provided.

        !!! tip "Data Prep Tip"
            Ensure `X` matches the `features_group` structure for accurate temporal modeling.
        """
        if self.features_group is None:
            raise ValueError("The features_group parameter must be provided.")

        return super().fit(X, y, *args, **kwargs)
