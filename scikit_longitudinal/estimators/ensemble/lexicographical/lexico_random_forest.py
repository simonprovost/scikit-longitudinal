# pylint: disable=too-many-arguments,invalid-name,signature-differs,no-member,R0801,R0901,R0902,W0221,R0401

from typing import List, Optional, Union

from sklearn.ensemble import RandomForestClassifier


class LexicoRandomForestClassifier(RandomForestClassifier):
    """
    Lexico Random Forest Classifier for longitudinal data classification.

    This classifier extends scikit-learn's `RandomForestClassifier` for longitudinal data by integrating a
    lexicographic optimisation approach within each tree of the forest, based on the premise that recent
    measurements are more predictive and relevant. Splits are evaluated with a bi-objective rule: the primary
    objective maximises the information-gain ratio (entropy criterion), and the secondary objective favours
    features from more recent waves whenever competing gain ratios are within `threshold_gain`. The ensemble
    aggregates these temporally-aware trees to reduce overfitting while preserving recency-driven decisions.

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
        class_weight (Optional[Union[dict, List[dict], str]], default=None):
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
        !!! example "Basic Usage"

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

        !!! example "Advanced: tuning threshold gain"

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
        class_weight: Optional[Union[dict, List[dict], str]] = None,
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
        # pylint: disable=C0415
        from scikit_longitudinal.estimators.trees import LexicoDecisionTreeClassifier

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

    def fit(self, X, y, sample_weight=None, *args, **kwargs):
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
        """
        if self.features_group is None:
            raise ValueError("The features_group parameter must be provided.")

        return super().fit(X, y, sample_weight=sample_weight, *args, **kwargs)

    def predict(self, X):
        """Predict class labels for the input samples.

        Inherited from scikit-learn's `RandomForestClassifier`. Each tree votes for a class and the
        class with the most votes is returned.

        Args:
            X (array-like of shape (n_samples, n_features)):
                Input samples.

        Returns:
            np.ndarray: Predicted class labels of shape `(n_samples,)`.
        """
        return super().predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for the input samples.

        Inherited from scikit-learn's `RandomForestClassifier`. Probabilities are the mean of the
        probabilistic predictions of the individual trees.

        Args:
            X (array-like of shape (n_samples, n_features)):
                Input samples.

        Returns:
            np.ndarray: Class probabilities of shape `(n_samples, n_classes)`, with columns ordered
            as in `self.classes_`.
        """
        return super().predict_proba(X)
