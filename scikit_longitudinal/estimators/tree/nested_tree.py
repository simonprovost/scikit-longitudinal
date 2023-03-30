from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import ray
from overrides import override
from sklearn.tree import DecisionTreeClassifier, plot_tree

from scikit_longitudinal.templates import CustomClassifierMixinEstimator


# pylint: disable=R0902,R0903,R0914,,too-many-arguments,invalid-name,signature-differs,no-member
class NestedTreesClassifier(CustomClassifierMixinEstimator):
    """Nested Trees Classifier for longitudinal data.

    The Nested Trees Classifier is a unique and innovative classification algorithm specifically designed for
    longitudinal datasets. It constructs a model similar to conventional decision tree algorithms but with a key
    difference: the nodes in the tree are not simple attribute-value tests but rather smaller, inner decision
    trees. This creates a two-layer structure with an outer tree containing inner decision trees at each node,
    taking full advantage of the longitudinal information present in the data. The outer decision use a custom
    decision tree that selects longitudinal attributes, which are groups of time-specific attributes, to build the
    tree. While the inner embedded decision tree is a Scikit Learn decision tree which works with subset based on
    the longitudinal attribute of the parent node.

    The main advantage of the Nested Trees algorithm is its longitudinal awareness. Unlike other approaches that
    flatten longitudinal data and treat different values of the same longitudinal attribute as independent
    attributes, this algorithm analyses each longitudinal attribute as a whole, preserving the longitudinal
    aspect of the data. This results in several benefits:

    1. Improved predictive accuracy: The algorithm can utilise the longitudinal information more effectively,
    potentially leading to better predictions.

    2. Enhanced model interpretability and attribute importance: The algorithm groups all temporal values of a
    longitudinal attribute together in an inner decision tree, making it easier to analyse the importance of a
    longitudinal attribute as a whole.

    3. Increased model acceptance by domain experts: A longitudinally-aware decision model that preserves the
    longitudinal nature of the data during model construction is more likely to be accepted by domain experts than
    a longitudinally unaware model.

    Parameters
    ----------
    group_features : List[List[int]]
        A list of lists, where each inner list contains the indices of features that
        correspond to a specific longitudinal attribute.

    max_outer_depth : int, optional (default=3)
        The maximum depth of the outer custom decision tree.

    max_inner_depth : int, optional (default=2)
        The maximum depth of the inner decision trees.

    min_outer_samples : int, optional (default=5)
        The minimum number of samples required to split an internal node in the outer
        decision tree.

    inner_estimator_hyperparameters : Dict[str, Any], optional (default=None)
        A dictionary of hyperparameters to be passed to the inner Scikit-learn
        decision tree estimators. If not provided, default hyperparameters will be
        used.

    save_nested_trees : bool, optional (default=False)
        If set to True, the nested trees structure plot will be saved, which may be useful
        for model interpretation and visualization.

    Attributes
    ----------
    root : Node, optional (default=None)
        The root node of the outer decision tree. Set to None upon initialisation, it
        will be updated during the model fitting process.

    Examples
    --------
    >>> from nested_trees import NestedTreesClassifier
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
    >>> y = np.array([0, 1, 0, 1])
    >>> group_features = [[0, 1], [2, 3]]
    >>> clf = NestedTreesClassifier(group_features=group_features)
    >>> clf.fit(X, y)
    >>> clf.predict(X)

    Notes
    -----
    For more information, see the following paper of the Nested Trees algorithm:

    Ovchinnik, S., Otero, F. and Freitas, A.A., 2022, April. Nested trees for
    longitudinal classification. In Proceedings of the 37th ACM/SIGAPP Symposium on
    Applied Computing (pp. 441-444). Vancouver

    Here is the initial Java implementation of the NESTED TREES algorithm:
    https://github.com/NestedTrees/NestedTrees

    See also
    ----------

    * CustomClassifierMixinEstimator: Base class for all Classifier Mixin estimators in scikit-learn that we
    customed so that the original scikit-learn "check_x_y" is performed all the time.

    Methods
    ---------

    fit(X, y)
         Fit the model according to the given training data.

    predict(X)
         Predict class for X.

    fit_predict(X, y)
         Fit the model according to the given training data and then predict the class on X.
    """

    # pylint: disable=too-many-arguments,invalid-name,signature-differs,no-member
    def __init__(
        self,
        group_features: List[List[int]],
        max_outer_depth: int = 3,
        max_inner_depth: int = 2,
        min_outer_samples: int = 5,
        inner_estimator_hyperparameters: Optional[Dict[str, Any]] = None,
        save_nested_trees: bool = False,
        parallel: bool = False,
        num_cpus: int = -1,
    ):
        self.group_features = group_features
        self.max_outer_depth = max_outer_depth
        self.max_inner_depth = max_inner_depth
        self.min_outer_samples = min_outer_samples
        self.inner_estimator_hyperparameters = inner_estimator_hyperparameters or {}
        self.save_nested_trees = save_nested_trees
        self.root = None
        self.parallel = parallel
        self.num_cpus = num_cpus

        if self.parallel and ray.is_initialized() is False:
            if num_cpus != -1:
                ray.init(num_cpus=self.num_cpus)
            else:
                ray.init()

        if max_outer_depth <= 0:
            raise ValueError("max_outer_depth must be greater than 0.")

        if max_inner_depth <= 0:
            raise ValueError("max_inner_depth must be greater than 0.")

        if min_outer_samples <= 0:
            raise ValueError("min_outer_samples must be greater than 0.")

    class Node:
        """A node in the outer decision tree of the Nested Trees Classifier.

        Each node in the outer tree contains an inner decision tree. Leaf nodes
        are associated with a class label, while non-leaf nodes have a decision
        criterion based on the inner decision tree.

        Parameters
        ----------

        is_leaf : bool
            Determines if the node is a leaf node. If True, the node represents a
            class label and has access to the tree associated with the last step it took to construct the leaf node;
            otherwise, it is an internal node with a decision criterion based on the inner decision tree.

        tree : DecisionTreeClassifier
            A Scikit-learn DecisionTreeClassifier instance representing the inner
            decision tree for this node.

        node_name : str
            A unique name for the node, used for visualization and debugging purposes.

        Attributes
        ----------
        is_leaf : bool
            Indicates whether the node is a leaf node.

        tree : DecisionTreeClassifier
            The inner decision tree associated with this node.

        children : List[Node]
            A list of child nodes of this node in the outer decision tree.

        children_map : Dict[str, Node]
            A dictionary mapping the name of each child node to the corresponding
            Node instance.

        node_name : str
            The unique name of this node.

        Examples
        --------
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> inner_tree = DecisionTreeClassifier()
        >>> node = Node(is_leaf=False, tree=inner_tree, node_name="dummy_node")
        """

        def __init__(self, is_leaf: bool, tree: DecisionTreeClassifier, node_name: str):
            if tree is None:
                raise ValueError("tree must be provided for (non-)leaf nodes.")

            if not node_name:
                raise ValueError("node_name must be a non-empty string.")

            self.is_leaf = is_leaf
            self.tree = tree
            self.children = []
            self.children_map = {}
            self.node_name = node_name

        def __str__(self):
            return self.node_name

    @override
    def _fit(self, X: np.ndarray, y: np.ndarray) -> "NestedTreesClassifier":
        """Fit the Nested Trees Classifier model according to the given training data.

        Parameters
        ----------
        X : np.ndarray
            The training input samples.
        y : np.ndarray
            The target values (class labels).

        Returns
        -------
        NestedTreesClassifier
            The fitted classifier.
        """
        self.root = self._build_outer_tree(X, y, 0, "outer_root")

        return self

    def _build_outer_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
        outer_node_name: str,
        tree: Optional[DecisionTreeClassifier] = None,
    ) -> "NestedTreesClassifier.Node":
        """Build the outer decision tree recursively.

        It starts at each node a competition between all possible inner decision trees, and
        the best inner decision tree is chosen to split the data. The gini impurity is used
        as the splitting criterion. The best inner decision tree leaves are then used as
        the children of the current node in the outer decision tree. Creating N (number of leaf nodes of the inner
        decision tree) children outer nodes in the outer decision tree.

        Max outer depth is the maximum depth of the outer decision tree. A minimum of 2 groups is required to
        go through the process of an outer node. The minimum number of samples required to split an outer node is
        determined by the min_outer_samples parameter.

        Parameters
        ----------
        X : np.ndarray
            The training input samples.
        y : np.ndarray
            The target values (class labels).
        depth : int
            The current depth of the outer tree being built.
        outer_node_name : str
            A unique name for the current node in the outer decision tree.
        tree : Optional[DecisionTreeClassifier], optional (default=None)
            The inner decision tree associated with this node.

        Returns
        -------
        NestedTreesClassifier.Node
            A node in the outer decision tree.
        """
        if depth == (self.max_outer_depth - 1) or len(self.group_features) < 2 or len(X) < self.min_outer_samples:
            return self.Node(is_leaf=True, tree=tree, node_name=outer_node_name)

        best_tree, best_split = self._find_best_tree_and_split(X, y, outer_node_name)

        if len(best_split) == 1:
            return self.Node(is_leaf=True, node_name=outer_node_name, tree=best_tree)

        node = self.Node(is_leaf=False, tree=best_tree, node_name=outer_node_name)
        self._add_children_to_node(node, best_split, depth)

        return node

    def _find_best_tree_and_split(
        self, X: np.ndarray, y: np.ndarray, outer_node_name: str
    ) -> Tuple[DecisionTreeClassifier, List[Tuple[np.ndarray, np.ndarray, int]]]:
        """Find the best inner decision tree and the split associated with it (i.e, the competition)

        Parameters
        ----------
        X : np.ndarray
            The training input samples.
        y : np.ndarray
            The target values (class labels).
        outer_node_name : str
            A unique name for the current node in the outer decision tree.

        Returns
        -------
        Tuple[DecisionTreeClassifier, List[Tuple[np.ndarray, np.ndarray, int]]]
            A tuple containing the best inner decision tree and the associated split.
        """

        # Start tasks to fit inner trees and calculate gini in parallel
        min_gini = float("inf")
        best_tree = None
        subset_X = None

        if self.parallel:
            tasks = [
                _fit_inner_tree_plus_calculate_gini_ray.remote(
                    X[:, group],
                    y,
                    i,
                    outer_node_name,
                    self.max_inner_depth,
                    self.inner_estimator_hyperparameters,
                    self.save_nested_trees,
                )
                for i, group in enumerate(self.group_features)
            ]
            results = ray.get(tasks)
            best_tree, _, min_gini, _, subset_X = min(results, key=lambda x: x[2])
        else:
            for i, group in enumerate(self.group_features):
                for i, group in enumerate(self.group_features):
                    subset_X = X[:, group]
                    tree, _, gini = _fit_inner_tree_and_calculate_gini(
                        subset_X,
                        y,
                        i,
                        outer_node_name,
                        self.max_inner_depth,
                        self.inner_estimator_hyperparameters,
                        self.save_nested_trees,
                    )

                    if gini < min_gini:
                        min_gini = gini
                        best_tree = tree

        best_split = self._create_split(X, subset_X, y, best_tree)
        return best_tree, best_split

    def _add_children_to_node(
        self, node: "NestedTreesClassifier.Node", best_split: List[Tuple[np.ndarray, np.ndarray, int]], depth: int
    ) -> None:
        """Add children to a node in the outer decision tree based on the best split.

        Note we had to map the leaf number of the inner decision tree to the child node in the outer decision tree
        because the leaf numbers are not necessarily sequential.

        Parameters
        ----------
        node : NestedTreesClassifier.Node
            The node in the outer decision tree to add children to.
        best_split : List[Tuple[np.ndarray, np.ndarray, int]]
            The best split of the data, represented as a list of tuples with (X subset, y subset, leaf number).
        depth : int
            The current depth of the node in the outer decision tree.
        """
        for i, (subset_X, subset_y, leaf_number) in enumerate(best_split):
            child_node_name = f"outer_{node.node_name}_d{depth + 1}_g{i}_l{leaf_number}"
            child_node = self._build_outer_tree(subset_X, subset_y, depth + 1, child_node_name, node.tree)
            node.children.append(child_node)
            node.children_map[leaf_number] = child_node

    def _create_split(
        self, X: np.ndarray, subset_X: np.ndarray, y: np.ndarray, tree: DecisionTreeClassifier
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """Create a split of the data based on the leaf nodes of the decision tree.

        Parameters
        ----------
        X : np.ndarray
            The original feature matrix.
        subset_X : np.ndarray
            The feature matrix for the current group.
        y : np.ndarray
            The target labels.
        tree : DecisionTreeClassifier
            The decision tree used for creating the split.

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray, int]]
            A list of tuples representing the split data, with each tuple containing:
            - X subset corresponding to a leaf node
            - y subset corresponding to a leaf node
            - Leaf number
        """
        leaves = tree.apply(subset_X)
        unique_leaves = np.unique(leaves)
        return [(X[leaves == leaf], y[leaves == leaf], leaf) for leaf in unique_leaves]

    @override
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        np.ndarray
            The predicted class labels for each input sample.
        """
        if self.root is None:
            raise ValueError("The classifier must be fitted before making predictions.")

        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x: np.ndarray) -> int:
        """Predict the class label for a single input sample.

        Parameters
        ----------
        x : np.ndarray
            The input sample.

        Returns
        -------
        int
            The predicted class label for the input sample.
        """
        node = self.root
        leaf_subset = None

        while not node.is_leaf:
            group = self._find_group(node.tree)
            subset_x = x[group]
            next_node_leaf_number = node.tree.apply([subset_x])[0]

            node = node.children_map[next_node_leaf_number]
            leaf_subset = subset_x
        return node.tree.predict(leaf_subset.reshape(1, -1))[0]

    def _find_group(self, tree: DecisionTreeClassifier) -> List[int]:
        """Find the group of features with non-zero feature importances in the given decision tree.

        The goal is to find the group of features that the decision tree used to make its predictions based
        on the group features of the class.

        Parameters
        ----------
        tree : DecisionTreeClassifier
            The decision tree to inspect for feature importances.

        Returns
        -------
        List[int]
            The group of features with non-zero feature importances.

        Raises
        ------
        ValueError
            If no group with non-zero feature importances is found.
        """

        for group in self.group_features:
            if np.any(tree.feature_importances_[group] != 0.0):
                return group
        raise ValueError("No group with non-zero feature importances found.")

    def print_nested_tree(
        self,
        node: Optional["NestedTreesClassifier.Node"] = None,
        depth: int = 0,
        prefix: str = "",
        parent_name: str = "",
    ) -> None:
        """Print the structure of the nested tree classifier.

        Parameters
        ----------
        node : Optional[NestedTreesClassifier.Node], optional (default=None)
            The current node in the outer decision tree. If None, start from the root node.
        depth : int, optional (default=0)
            The current depth of the node in the outer decision tree.
        prefix : str, optional (default="")
            A string to prepend before the node's name in the output.
        parent_name : str, optional (default="")
            The name of the parent node in the outer decision tree.
        """
        if node is None:
            node = self.root

        node_name_parts = node.node_name.split("_")
        unique_node_name_parts = self._remove_consecutive_duplicates(node_name_parts)
        node_name = "_".join(unique_node_name_parts)

        if parent_name:
            node_name = node_name.replace(f"{parent_name}_", "")

        if node.is_leaf:
            print(f"{prefix}* Leaf {depth}: {node_name}")
        else:
            print(f"{prefix}* Node {depth}: {node_name}")
            for child in node.children:
                self.print_nested_tree(child, depth + 1, f"{prefix}  ", node_name)

    def _remove_consecutive_duplicates(self, values: List[str]) -> List[str]:
        """Remove consecutive duplicates in a list of strings.

        Examples were taken from the following string node printed from the print_nested_tree method:

        "outer_node_outer_node_..." -> "outer_node_..."

        Parameters
        ----------
        values : List[str]
            The list of strings to process.

        Returns
        -------
        List[str]
            The list of strings with consecutive duplicates removed.
        """
        result = []
        prev = None
        for item in values:
            if item != prev:
                result.append(item)
            prev = item
        return result


@ray.remote
def _fit_inner_tree_plus_calculate_gini_ray(
    subset_X: np.ndarray,
    y: np.ndarray,
    group_index: int,
    outer_node_name: str,
    max_inner_depth: int,
    inner_estimator_hyperparameters: Dict[str, Any],
    save_nested_trees: bool,
) -> Tuple[DecisionTreeClassifier, Any, float, np.ndarray]:
    """Copy of _fit_inner_tree_plus_calculate_gini to be used with Ray parallelization."""

    tree, y_pred, gini = _fit_inner_tree_and_calculate_gini(
        subset_X, y, group_index, outer_node_name, max_inner_depth, inner_estimator_hyperparameters, save_nested_trees
    )
    return tree, outer_node_name, gini, y_pred, subset_X


def _fit_inner_tree_and_calculate_gini(
    subset_X: np.ndarray,
    y: np.ndarray,
    group_index: int,
    outer_node_name: str,
    max_inner_depth: int,
    inner_estimator_hyperparameters: Dict[str, Any],
    save_nested_trees: bool,
) -> Tuple[DecisionTreeClassifier, np.ndarray, float]:
    """Fit an inner decision tree to a subset of data and calculate the Gini impurity of its predictions.

    Parameters
    ----------
    subset_X : np.ndarray
        The training input samples for a specific group of features.
    y : np.ndarray
        The target values (class labels).
    group_index : int
        The index of the current group of features.
    outer_node_name : str
        A unique name for the current node in the outer decision tree.

    Returns
    -------
    Tuple[DecisionTreeClassifier, np.ndarray, float]
        A tuple containing the fitted inner decision tree, the predicted labels, and the Gini impurity.
    """
    tree = DecisionTreeClassifier(max_depth=max_inner_depth, **inner_estimator_hyperparameters)
    tree.fit(subset_X, y)
    if save_nested_trees:
        _save_inner_tree(tree, f"inner_tree_{outer_node_name}_group_{group_index}.png")
    y_pred = tree.predict(subset_X)
    gini = _calculate_gini(y, y_pred)
    return tree, y_pred, gini


def _calculate_gini(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Gini impurity of the predictions.

    Formula taken from the original implemmentation of the Nested Trees algorithm:
    https://github.com/NestedTrees/NestedTrees/blob/main/src/ModelEvaluator.java#L105

    Parameters
    ----------
    y : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.

    Returns
    -------
    float
        The Gini impurity of the predictions.
    """
    total_count = len(y)
    gini = 0.0

    for class_value in np.unique(y):
        class_indices = np.where(y_pred == class_value)
        class_subset = y[class_indices]
        class_count = len(class_subset)

        if class_count > 0:
            _, counts = np.unique(class_subset, return_counts=True)
            class_gini = 1 - sum((count / class_count) ** 2 for count in counts)
            gini += class_gini * (class_count / total_count)

    return gini


def _save_inner_tree(tree: DecisionTreeClassifier, filename: str) -> None:
    """Save a visual representation of the given decision tree as an image file.

    Parameters
    ----------
    tree : DecisionTreeClassifier
        The decision tree to visualize.
    filename : str
        The name of the file to save the image as.
    """
    plt.figure(figsize=(10, 10))
    plot_tree(tree, filled=True)
    plt.savefig(filename)
    plt.close()
