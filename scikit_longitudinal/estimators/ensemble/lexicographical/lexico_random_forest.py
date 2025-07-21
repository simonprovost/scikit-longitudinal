# pylint: disable=too-many-arguments,invalid-name,signature-differs,no-member,R0801,R0901,R0902,W0221,R0401

from typing import List, Optional, Union

from sklearn.ensemble import RandomForestClassifier


class LexicoRandomForestClassifier(RandomForestClassifier):
    """
    Lexico Random Forest Classifier for longitudinal data classification.

    The Lexico Random Forest Classifier is an advanced ensemble algorithm tailored for longitudinal data analysis. It
    extends the traditional Random Forest by integrating a lexicographic optimization approach within each decision tree,
    prioritizing more recent data points (waves) for splits. This is based on the premise that recent measurements are more
    predictive and relevant, making it ideal for applications like medical studies or time-series classification. The
    implementation leverages a Cython-optimized fork of scikit-learn's decision tree for enhanced efficiency.

    !!! tip "Why Use LexicoRandomForestClassifier?"
        This classifier excels with longitudinal datasets where temporal recency is key. By combining lexicographic
        optimization with the ensemble strength of random forests, it captures evolving patterns while minimizing
        overfitting—perfect for robust predictive modeling.

    !!! question "How Does Lexicographic Optimization Work?"
        Each tree in the forest employs a bi-objective split selection strategy:

        1. **Primary**: Maximize the information gain ratio using the "entropy" criterion.
        2. **Secondary**: Favor features from more recent waves when gain ratios are similar (within `threshold_gain`).

        This ensures both statistical purity and temporal relevance are optimized, with the ensemble aggregating these
        decisions for improved accuracy.

    !!! note "Performance Boost with Cython"
        The splitter (`node_lexicoRF_split`) is optimized in Cython for faster computation. See the
        [Cython implementation](https://github.com/simonprovost/scikit-lexicographical-trees/blob/21443b9dce51434b3198ccabac8bafc4698ce953/sklearn/tree/_splitter.pyx#L695)
        for details.

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
        n_estimators (int, default=100):
            The number of trees in the forest.
        threshold_gain (float, default=0.0015):
            Threshold for comparing gain ratios during split selection. Lower values enforce stricter recency preference;
            higher values allow more flexibility.
        features_group (List[List[int]], optional):
            Temporal matrix of feature indices for longitudinal attributes, ordered by recency. Required for longitudinal
            functionality.
        max_depth (Optional[int], default=None):
            Maximum depth of each tree.
        min_samples_split (int, default=2):
            Minimum samples required to split an internal node.
        min_samples_leaf (int, default=1):
            Minimum samples required at a leaf node.
        min_weight_fraction_leaf (float, default=0.0):
            Minimum weighted fraction of total sample weight at a leaf.
        max_features (Optional[Union[int, str]], default="sqrt"):
            Number of features to consider for splits (e.g., "sqrt", "log2", int).
        random_state (Optional[int], default=None):
            Seed for random number generation.
        max_leaf_nodes (Optional[int], default=None):
            Maximum number of leaf nodes per tree.
        min_impurity_decrease (float, default=0.0):
            Minimum impurity decrease required for a split.
        class_weight (Optional[Union[dict, str]], default=None):
            Class weights (e.g., `{class_label: weight}` or "balanced").
        ccp_alpha (float, default=0.0):
            Complexity parameter for pruning; non-negative.
        **kwargs:
            Additional arguments for `RandomForestClassifier`.

    Attributes:
        classes_ (ndarray of shape (n_classes,)):
            The class labels.
        n_classes_ (int):
            Number of classes.
        n_features_ (int):
            Number of features when fit is performed.
        n_outputs_ (int):
            Number of outputs (fixed to 1).
        feature_importances_ (ndarray of shape (n_features,)):
            Impurity-based feature importances.
        max_features_ (int):
            Inferred value of `max_features`.
        estimators_ (list of LexicoDecisionTreeClassifier):
            Fitted tree ensemble.

    Examples:
        !!! example "Basic Usage with Dummy Longitudinal Data"

            ```python
            from sklearn.metrics import accuracy_score
            from scikit_longitudinal.estimators.ensemble import LexicoRandomForestClassifier
            import numpy as np
            from scikit_longitudinal.data_preparation import LongitudinalDataset

            # Load dataset
            dataset = LongitudinalDataset('./stroke_longitudinal.csv')
            dataset.load_data()
            dataset.load_target(target_column="stroke_w2")
            dataset.setup_features_group("elsa")
            dataset.load_train_test_split(test_size=0.2, random_state=42)

            clf = LexicoRandomForestClassifier(features_group=dataset.feature_groups())
            clf.fit(dataset.X_train, dataset.y_train)
            y_pred = clf.predict(dataset.X_test)
            print(f"Accuracy: {accuracy_score(dataset.y_test, y_pred)}")
            ```

        !!! example "Tuning Threshold Gain"

            ```python
            from sklearn.metrics import accuracy_score
            from scikit_longitudinal.estimators.ensemble import LexicoRandomForestClassifier
            import numpy as np
            from scikit_longitudinal.data_preparation import LongitudinalDataset

            # Load dataset
            dataset = LongitudinalDataset('./stroke_longitudinal.csv')
            dataset.load_data()
            dataset.load_target(target_column="stroke_w2")
            dataset.setup_features_group("elsa")
            dataset.load_train_test_split(test_size=0.2, random_state=42)

            clf = LexicoRandomForestClassifier(
                features_group=dataset.feature_groups(),
                threshold_gain=0.001 # Change this value to tune the model
            )
            clf.fit(dataset.X_train, dataset.y_train)
            y_pred = clf.predict(dataset.X_test)
            print(f"Accuracy: {accuracy_score(dataset.y_test, y_pred)}")

            clf = LexicoRandomForestClassifier(threshold_gain=0.001, features_group=[[0, 1], [2, 3]])
            clf.fit(X, y)
            y_pred = clf.predict(X)
            print(f"Accuracy: {accuracy_score(y, y_pred)}")
            ```

            !!! tip "Hyperparameter Tuning"
                Use `GridSearchCV` or `RandomizedSearchCV` from `sklearn.model_selection` to optimize `threshold_gain`
                and other hyperparameters. For example:

                ```python
                from sklearn.model_selection import GridSearchCV

                # ... Similar setup as above ...

                param_grid = {
                    'threshold_gain': [0.001, 0.002, 0.003],
                    'n_estimators': [50, 100, 200],
                }
                grid_search = GridSearchCV(
                    LexicoRandomForestClassifier(
                        # features_group = ... Same as previous example ...
                    )
                    param_grid,
                    cv=5,
                    scoring='accuracy',
                )
                grid_search.fit(X, y)
                print(f"Best parameters: {grid_search.best_params_}")
                ```

    Notes:
        - **References**:

          - Ribeiro, C. and Freitas, A., 2020. "A new random forest method for longitudinal data classification using a
            lexicographic bi-objective approach." *2020 IEEE Symposium Series on Computational Intelligence (SSCI)*,
            pp. 806-813.
          - Ribeiro, C. and Freitas, A.A., 2024. "A lexicographic optimisation approach to promote more recent features
            on longitudinal decision-tree-based classifiers." *Artificial Intelligence Review*, 57(4), p.84.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        threshold_gain: float = 0.0015,
        features_group: List[List[int]] = None,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[Union[int, str]] = "sqrt",
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        class_weight: Optional[str] = None,
        ccp_alpha: float = 0.0,
        random_state: int = None,
        **kwargs,
    ):
        self.threshold_gain = threshold_gain
        self.features_group = features_group
        self.criterion = "entropy"
        self.splitter = "lexicoRF"
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.estimator_ = None

        super().__init__(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            **kwargs,
        )

    def _validate_estimator(self):
        from scikit_longitudinal.estimators.trees import LexicoDecisionTreeClassifier  # pylint: disable=C0415

        self.estimator_ = LexicoDecisionTreeClassifier(
            threshold_gain=self.threshold_gain,
            features_group=self.features_group,
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
        )

    def fit(self, X, y, *args, **kwargs):
        """
        Fit the Lexico Random Forest Classifier to the training data.

        Trains the classifier by constructing multiple LexicoDecisionTreeClassifiers, each utilizing the `features_group`
        for lexicographic optimization tailored to longitudinal data.

        Args:
            X (array-like of shape (n_samples, n_features)):
                Training input samples.
            y (array-like of shape (n_samples,)):
                Target values (class labels).
            *args:
                Additional positional arguments for the superclass `fit`.
            **kwargs:
                Additional keyword arguments for the superclass `fit`.

        Returns:
            self: Fitted classifier instance.

        Raises:
            ValueError: If `features_group` is not provided.

        !!! tip "Tuning Tip"
            Adjust `n_estimators` and `threshold_gain` to balance accuracy and computation time. Start with defaults
            and refine based on your dataset.

        !!! note
            Ensure `features_group` accurately maps your data's temporal structure for optimal performance.
        """
        if self.features_group is None:
            raise ValueError("The features_group parameter must be provided.")

        return super().fit(X, y, *args, **kwargs)
