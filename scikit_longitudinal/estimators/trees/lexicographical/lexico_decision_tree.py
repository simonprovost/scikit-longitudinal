# pylint: disable=too-many-arguments,invalid-name,signature-differs,no-member,R0801

from typing import List, Optional, Union

from sklearn.tree import DecisionTreeClassifier


class LexicoDecisionTreeClassifier(DecisionTreeClassifier):
    """
    Lexico Decision Tree Classifier for longitudinal data classification.

    This classifier extends scikit-learn's `DecisionTreeClassifier` for longitudinal data by integrating a
    lexicographic optimisation approach that prioritises more recent waves during split selection, based on the
    premise that recent measurements are more predictive and relevant. Splits are evaluated with a bi-objective
    rule: the primary objective maximises the information-gain ratio (entropy criterion), and the secondary
    objective favours features from more recent waves whenever competing gain ratios are within `threshold_gain`.

    Args:
        threshold_gain (float, default=0.0015):
            The threshold value for comparing gain ratios of features during tree construction. A lower value makes
            the algorithm more selective in choosing recent features, requiring gain ratios to be closer for recency
            to take precedence. A higher value allows larger differences in gain ratios while still considering recency.
        features_group (List[List[int]], optional):
            A list of lists where each inner list contains indices of features corresponding to a specific longitudinal
            attribute across different waves. The order within each inner list reflects the temporal sequence, with the
            first element being the oldest wave and the last being the most recent. For example, `[[0,1],[2,3]]` indicates
            two longitudinal attributes, each with two waves (e.g., 0: oldest, 1: recent; 2: oldest, 3: recent).
        criterion (str, default="entropy"):
            The function to measure the quality of a split. Fixed to "entropy" for this algorithm; do not change.
        splitter (str, default="lexicoRF"):
            The strategy used to choose the split at each node. Fixed to "lexicoRF" for this algorithm; do not change.
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
        class_weight (Optional[Union[dict, List[dict], str]], default=None):
            Weights associated with classes in the form `{class_label: weight}` or "balanced".
        ccp_alpha (float, default=0.0):
            Complexity parameter used for Minimal Cost-Complexity Pruning. Must be non-negative.
        store_leaf_values (bool, default=False):
            Whether to store leaf values during tree construction.
        monotonic_cst (Optional[List[int]], default=None):
            Monotonic constraints for features (1 for increasing, -1 for decreasing, 0 for no constraint).

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
        Below are examples demonstrating the usage of the `LexicoDecisionTreeClassifier` class.

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
            from scikit_longitudinal.estimators.trees import LexicoDecisionTreeClassifier

            # Load dataset
            X, y = load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Define features_group (example for illustration; adjust based on actual longitudinal structure)
            features_group = [[0, 1], [2, 3]]

            # Initialize and fit the classifier
            clf = LexicoDecisionTreeClassifier(threshold_gain=0.1, features_group=features_group)
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
            from scikit_longitudinal.estimators.trees import LexicoDecisionTreeClassifier
            from scikit_longitudinal.data_preparation import LongitudinalDataset
            from scikit_longitudinal.data_preparation import MerWavTimePlus

            # Load dataset
            dataset = LongitudinalDataset('./stroke_longitudinal.csv')
            dataset.load_data()
            dataset.load_target(target_column="stroke_w2")
            dataset.setup_features_group("elsa")
            dataset.load_train_test_split(test_size=0.2, random_state=42)

            # Define pipeline steps with LexicoDecisionTreeClassifier
            steps = [
                ('MerWavTime Plus', MerWavTimePlus()), # Recall, a pipeline is at least two steps and the first one being a Data Transformation step. Here as we use a Longitudinal classifier, we need to use MerWavTimePlus, retaining the temporal dependency.
                ('classifier', LexicoDecisionTreeClassifier(features_group=dataset.feature_groups()))
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

    def __init__(
        self,
        threshold_gain: float = 0.0015,
        features_group: List[List[int]] = None,
        criterion: str = "entropy",  # Do not change this value
        splitter: str = "lexicoRF",  # Do not change this value
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[Union[int, str]] = None,
        random_state: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        class_weight: Optional[Union[dict, List[dict], str]] = None,
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
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            store_leaf_values=store_leaf_values,
            monotonic_cst=monotonic_cst,
        )

    def fit(self, X, y, sample_weight=None, *args, **kwargs):
        """
        Fit the Lexico Decision Tree Classifier to the training data.

        This method trains the classifier using the provided training data and labels. It requires the `features_group`
        parameter to be set, as it is essential for handling longitudinal data.

        Args:
            X (array-like of shape (n_samples, n_features)):
                The training input samples.
            y (array-like of shape (n_samples,)):
                The target values (class labels).
            *args:
                Additional positional arguments passed to the superclass `fit` method.
            **kwargs:
                Additional keyword arguments passed to the superclass `fit` method.

        Returns:
            LexicoDecisionTreeClassifier:
                The fitted classifier instance.

        Raises:
            ValueError:
                If `features_group` is not provided, as it is required for longitudinal functionality.
        """
        if self.features_group is None:
            raise ValueError("The features_group parameter must be provided.")

        return super().fit(X, y, sample_weight=sample_weight, *args, **kwargs)

    def predict(self, X, check_input=True):
        """Predict class labels for the input samples.

        Inherited from scikit-learn's `DecisionTreeClassifier`. The Lexico tree only customises
        split selection at fit time; prediction is the standard tree-traversal routine.

        Args:
            X (array-like of shape (n_samples, n_features)):
                Input samples.
            check_input (bool, default=True):
                Allow to bypass input validation. Forwarded to scikit-learn.

        Returns:
            np.ndarray: Predicted class labels of shape `(n_samples,)`.
        """
        return super().predict(X, check_input=check_input)

    def predict_proba(self, X, check_input=True):
        """Predict class probabilities for the input samples.

        Inherited from scikit-learn's `DecisionTreeClassifier`. Probabilities are the fraction of
        training samples of each class in the leaf reached by each input sample.

        Args:
            X (array-like of shape (n_samples, n_features)):
                Input samples.
            check_input (bool, default=True):
                Allow to bypass input validation. Forwarded to scikit-learn.

        Returns:
            np.ndarray: Class probabilities of shape `(n_samples, n_classes)`, with columns ordered
            as in `self.classes_`.
        """
        return super().predict_proba(X, check_input=check_input)
