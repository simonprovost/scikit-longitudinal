# pylint: disable=too-many-arguments,invalid-name,signature-differs,no-member,R0801

from typing import List, Optional, Union

from sklearn_fork.tree import DecisionTreeClassifier


class LexicoDecisionTreeClassifier(DecisionTreeClassifier):
    """LexicoDecisionTree.

    This implementation provides the LexicoDecisionTree, an adaptation of the sklearn decision tree algorithm for
    longitudinal data classification. The lexicographic approach considers both information gain ratios and time
    points (wave ids) when selecting split features in a decision-tree node, optimising the primary objective of
    maximising the gain ratio and the secondary objective of maximising the time-index (wave id) of the features.
    The algorithm aims to improve predictive performance by favoring more recent information in longitudinal
    datasets. Further information will be available in the Cython adaptation of the algorithm, available at
    /scikit-longitudinal/scikit-learn/sklearn/tree/_splitter.pyx function ``node_lexicoRF_split''.

    However, here is how features_group works:

    Consider you have a dataset with 4 features, and you want to split them into 2 groups, where the first group
    contains the first 2 features, and the second group contains the last 2 features. A real-world example would be
    smoke and cholesterol, each with two waves, and you want to split them into two groups, one with the first
    longitudinal attribute (smoke) and the other with the second longitudinal attribute (cholesterol). In this case,
    you would pass the following list of lists as the features_group parameter:

    [[0,1],[2,3]], where 0 and 1 are the indices of the first longitudinal attribute, and 2 and 3 are the indices
    of the second longitudinal attribute. So 0, is smoke wave 1, 1 is smoke wave 2, 2 is cholesterol wave 1, and 3
    is cholesterol wave 2. Hence, the algorithm can deal with the feature recentness, i.e., the first element of
    the inner lists are older, and the farther the element is from the first element, the more recent it is.

    Args:
        threshold_gain : float
            The threshold value for comparing gain ratios of features during the decision tree construction.
        features_group : List[List[int]]
            A list of lists, where each inner list contains the indices of features that
            correspond to a specific longitudinal attribute.
        criterion : str, optional (default="entropy")
            The function to measure the quality of a split. Do not change this value.
        splitter : str, optional (default="lexicoRF")
            The strategy used to choose the split at each node. Do not change this value.
        max_depth : int, optional (default=None)
            The maximum depth of the tree.
        min_samples_split : int, optional (default=2)
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int, optional (default=1)
            The minimum number of samples required to be at a leaf node.
        min_weight_fraction_leaf : float, optional (default=0.)
            The minimum weighted fraction of the sum total of weights required to be at a leaf node.
        max_features : int, optional (default=None)
            The number of features to consider when looking for the best split.
        random_state : int, optional (default=None)
            The seed used by the random number generator.
        max_leaf_nodes : int, optional (default=None)
            The maximum number of leaf nodes in the tree.
        min_impurity_decrease : float, optional (default=0.)
            The minimum impurity decrease required for a node to be split.
        class_weight : str, optional (default=None)
            Weights associated with classes in the form of {class_label: weight}.
        ccp_alpha : float, optional (default=0.0)
            Complexity parameter used for Minimal Cost-Complexity Pruning.

    Attributes:
        classes_ : ndarray of shape (n_classes,)
            The classes labels (single output problem).
        n_classes_ : int
            The number of classes (single output problem).
        n_features_ : int
            The number of features when fit is performed.
        n_outputs_ : int
            The number of outputs when fit is performed.
        feature_importances_ : ndarray of shape (n_features,)
            The impurity-based feature importances.
        max_features_ : int
            The inferred value of max_features.
        tree_ : Tree object
            The underlying Tree object.

    Examples:
        >>> from sklearn_fork.datasets import load_iris
        >>> from sklearn_fork.model_selection import train_test_split
        >>> from sklearn_fork.metrics import accuracy_score
        >>> from scikit_longitudinal.estimators.tree import LexicoDecisionTreeClassifier
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        >>> clf = LexicoDecisionTreeClassifier(threshold_gain=0.1, features_group=[[0,1],[2,3]])
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
        >>> accuracy_score(y_test, y_pred)
        0.9666666666666667 (dummy example with dummy non-longitudinal data)

    Notes:
        For more information, please refer to the following paper:

        Ribeiro, C. and Freitas, A., 2020, December. A new random forest method for longitudinal data
        classification using a lexicographic bi-objective approach. In 2020 IEEE Symposium Series on

    """

    def __init__(
        self,
        threshold_gain: float = 0.10,
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
        class_weight: Optional[str] = None,
        ccp_alpha: float = 0.0,
    ):
        self.threshold_gain = threshold_gain
        self.features_group = features_group
        super().__init__(
            criterion=criterion,
            threshold_gain=threshold_gain,
            features_group=features_group,
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
        )
