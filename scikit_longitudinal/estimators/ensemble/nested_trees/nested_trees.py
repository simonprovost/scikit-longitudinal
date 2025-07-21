from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import ray
from overrides import override
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels

from scikit_longitudinal.data_preparation.longitudinal_dataset import clean_padding
from scikit_longitudinal.estimators.ensemble.nested_trees.utils import (
    _fit_inner_tree_and_calculate_gini,
    _fit_inner_tree_plus_calculate_gini_ray,
    _remove_consecutive_duplicates,
)
from scikit_longitudinal.templates import CustomClassifierMixinEstimator


# pylint: disable=R0902,R0903,R0914,,too-many-arguments,invalid-name,signature-differs,no-member
class NestedTreesClassifier(CustomClassifierMixinEstimator):
    """
    Nested Trees Classifier for longitudinal data classification.

    The Nested Trees Classifier is a unique and innovative algorithm tailored for longitudinal datasets. It enhances
    traditional decision tree methods by embedding smaller decision trees within the nodes of a primary tree structure,
    optimally leveraging the temporal information inherent in longitudinal data. This hierarchical approach excels at
    capturing complex temporal patterns and dependencies.

    !!! info "Structure Overview"
        The outer tree uses a custom algorithm to select longitudinal attributes (groups of time-specific features).
        Each node hosts an inner `DecisionTreeClassifier` from scikit-learn, partitioning data based on the selected
        attribute, creating a nested decision-making process.

        We highly  recommend to read the paper to better understand the primitive.

    !!! question "Feature Groups and Non-Longitudinal Features"
        Two key attributes define the temporal structure:

        - **features_group**: A list of lists, each sublist containing indices of a longitudinal attribute's waves,
          ordered from oldest to most recent (e.g., `[[0,1], [2,3]]` for two attributes with two waves each).
        - **non_longitudinal_features**: Indices of static features (not used in temporal modeling but included in splits).

        Accurate configuration is essential. See the [Temporal Dependency Guide](https://scikit-longitudinal.readthedocs.io/latest/tutorials/temporal_dependency/).

    Args:
        features_group (List[List[int]], optional):
            Temporal matrix of feature indices for longitudinal attributes. Required for longitudinal functionality.
        non_longitudinal_features (List[Union[int, str]], optional):
            Indices of static, non-temporal features. Defaults to None.
        max_outer_depth (int, optional):
            Maximum depth of the outer decision tree. Defaults to 3.
        max_inner_depth (int, optional):
            Maximum depth of inner decision trees. Defaults to 2.
        min_outer_samples (int, optional):
            Minimum samples required to split an outer node. Defaults to 5.
        inner_estimator_hyperparameters (Optional[Dict[str, Any]], optional):
            Hyperparameters for inner decision trees. Defaults to None.
        save_nested_trees (bool, optional):
            If True, saves visualizations of the nested structure. Defaults to False.
        parallel (bool, optional):
            Enables parallel processing for fitting inner trees. Defaults to False.
        num_cpus (int, optional):
            Number of CPUs for parallel processing (-1 uses all available). Defaults to -1.

    Attributes:
        root (Node, optional):
            Root node of the outer tree, initialized as None and set during fitting.
        classes_ (np.ndarray):
            Unique class labels, set during fitting.

    Examples:
        !!! example "Basic Usage with Dummy Longitudinal Data"

            ```python
            from sklearn.metrics import accuracy_score
            from scikit_longitudinal.estimators.ensemble import NestedTreesClassifier
            import numpy as np
            from scikit_longitudinal.data_preparation import LongitudinalDataset

            # Load dataset
            dataset = LongitudinalDataset('./stroke_longitudinal.csv')
            dataset.load_data()
            dataset.load_target(target_column="stroke_w2")
            dataset.setup_features_group("elsa")
            dataset.load_train_test_split(test_size=0.2, random_state=42)

            clf = NestedTreesClassifier(features_group=dataset.feature_groups())
            clf.fit(dataset.X_train, dataset.y_train)
            y_pred = clf.predict(dataset.X_test)
            print(f"Accuracy: {accuracy_score(dataset.y_test, y_pred)}")
            ```

        !!! example "Customizing Inner Tree Hyperparameters"

            ```python
            # ... Similar setup as above ...

            inner_params = {'criterion': 'gini', 'max_depth': 3}
            clf = NestedTreesClassifier(
                features_group=features_group,
                non_longitudinal_features=non_longitudinal_features,
                inner_estimator_hyperparameters=inner_params
            )
            clf.fit(X, y)

            # ... Similar prediction and evaluation as above ...
            ```

    Notes:
        - Requires accurate `features_group` and `non_longitudinal_features` setup for optimal temporal modeling.
        - References: Ovchinnik, S., Otero, F., & Freitas, A.A. (2022). *Nested trees for longitudinal classification.*
          ACM/SIGAPP Symposium on Applied Computing, 441-444.
        - Original Java implementation: [Nested Trees GitHub](https://github.com/NestedTrees/NestedTrees).
    """

    # pylint: disable=too-many-arguments,invalid-name,signature-differs,no-member
    def __init__(
        self,
        features_group: List[List[int]] = None,
        non_longitudinal_features: List[Union[int, str]] = None,
        max_outer_depth: int = 3,
        max_inner_depth: int = 2,
        min_outer_samples: int = 5,
        inner_estimator_hyperparameters: Optional[Dict[str, Any]] = None,
        save_nested_trees: bool = False,
        parallel: bool = False,
        num_cpus: int = -1,
    ):
        self.features_group = features_group
        self.non_longitudinal_features = non_longitudinal_features
        self.max_outer_depth = max_outer_depth
        self.max_inner_depth = max_inner_depth
        self.min_outer_samples = min_outer_samples
        self.inner_estimator_hyperparameters = inner_estimator_hyperparameters
        self.save_nested_trees = save_nested_trees
        self.root = None
        self.parallel = parallel
        self.num_cpus = num_cpus
        self.classes_ = None

        if self.parallel and ray.is_initialized() is False:  # pragma: no cover
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

        Args:
            is_leaf (bool):
                Determines if the node is a leaf node. If True, the node represents a class label and has access to the
                tree associated with the last step it took to construct the leaf node; otherwise, it is an internal node
                with a decision criterion based on the inner decision tree.
            tree (DecisionTreeClassifier):
                A Scikit-learn DecisionTreeClassifier instance representing the inner decision tree for this node.
            node_name (str):
                A unique name for the node, used for visualization and debugging purposes.
            group (Optional[List[int]]):
                A list of feature group indices used to split the data in this node.
                Defaults to None.

        Attributes:
            is_leaf (bool):
                Indicates whether the node is a leaf node.
            tree (DecisionTreeClassifier):
                The inner decision tree associated with this node.
            children (List[Node]):
                A list of child nodes of this node in the outer decision tree.
            children_map (Dict[str, Node]):
                A dictionary mapping the name of each child node to the corresponding Node instance.
            node_name (str):
                The unique name of this node.
            group (Optional[List[int]]):
                A list of feature group indices used to split the data in this node.

        Raises:
            ValueError:
                If tree is not provided, or if node_name is an empty string.

        Examples:
            >>> from sklearn.tree import DecisionTreeClassifier
            >>> inner_tree = DecisionTreeClassifier()
            >>> node = Node(is_leaf=False, tree=inner_tree, node_name="dummy_node")

        """

        def __init__(
            self,
            is_leaf: bool,
            tree: DecisionTreeClassifier,
            node_name: str,
            group: Optional[List[int]] = None,
        ):
            if tree is None:
                raise ValueError("tree must be provided for (non-)leaf nodes.")

            if not node_name:
                raise ValueError("node_name must be a non-empty string.")

            self.is_leaf = is_leaf
            self.tree = tree
            self.children = []
            self.children_map = {}
            self.node_name = node_name
            self.group = group

        def __str__(self):
            return self.node_name

    @override
    def _fit(self, X: np.ndarray, y: np.ndarray) -> "NestedTreesClassifier":
        """Fit the classifier to the training data.

        Builds the nested tree structure recursively, integrating longitudinal and non-longitudinal features.

        Args:
            X (np.ndarray): Training input samples.
            y (np.ndarray): Target class labels.

        Returns:
            NestedTreesClassifier: Fitted classifier instance.

        Raises:
            ValueError: If `features_group` has fewer than 2 groups.

        !!! tip "Tuning Advice"
            Increase `max_outer_depth` for complex datasets, but monitor for overfitting with validation data.
        """
        if self.non_longitudinal_features is not None:
            self.features_group.append(self.non_longitudinal_features)
        if self.features_group is not None:
            self.features_group = clean_padding(self.features_group)

        if not self.features_group or len(self.features_group) <= 1:
            raise ValueError("features_group must be greater than 1.")
        if self.inner_estimator_hyperparameters is None:
            self.inner_estimator_hyperparameters = {}
        if self.classes_ is None:
            self.classes_ = unique_labels(y)
        self.root = self._build_outer_tree(X, y, 0, "outer_root")
        return self

    @override
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input samples.

        Traverses the nested tree structure to assign labels based on outer and inner tree decisions.

        Args:
            X (np.ndarray): Input samples.

        Returns:
            np.ndarray: Predicted class labels.

        Raises:
            ValueError: If the classifier isn’t fitted (root is None).

        !!! tip "Quick Predictions"
            After fitting, use this method to generate predictions efficiently leveraging the nested structure.
        """
        if self.root is None:
            raise ValueError("The classifier must be fitted before making predictions.")
        return np.array([self._predict_single(x) for x in X])

    @override
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for input samples.

        Provides probability estimates by traversing the nested structure.

        Args:
            X (np.ndarray): Input samples.

        Returns:
            np.ndarray: Predicted class probabilities.

        Raises:
            ValueError: If the classifier isn’t fitted or probability shapes are inconsistent.

        !!! question "When to Use Probabilities?"
            Use `predict_proba` instead of `predict` when you need confidence scores or custom thresholds, such as in
            medical diagnostics.
        """
        if self.root is None:
            raise ValueError("The classifier must be fitted before making predictions.")

        result = []
        first_run = True
        expected_shape = None
        for x in X:
            probas = self._predict_proba_single(x)
            if probas.shape == (1, 1):
                probas = np.array([[probas[0][0], 1.0 - probas[0][0]]])
            probas = probas.flatten()
            if first_run:
                expected_shape = probas.shape
                first_run = False
            elif probas.shape != expected_shape:
                raise ValueError(f"Unexpected shape for predict_proba: {probas.shape}, expected: {expected_shape}")
            result.append(probas)

        return np.array(result)

    def _build_outer_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
        outer_node_name: str,
        tree: Optional[DecisionTreeClassifier] = None,
        group: Optional[List[int]] = None,
    ) -> "NestedTreesClassifier.Node":
        """Build the outer decision tree recursively.

        The method starts at each node, competing between all possible inner decision trees.
        The best inner decision tree is chosen to split the data based on Gini impurity.
        The best inner decision tree's leaves are then used as the children of the current
        node in the outer decision tree, creating N (number of leaf nodes of the inner decision tree)
        children outer nodes in the outer decision tree.

        The max_outer_depth parameter determines the maximum depth of the outer decision tree.
        A minimum of 2 groups is required to process an outer node. The min_outer_samples parameter sets the
        minimum number of samples required to split an outer node.

        Args:
            X (np.ndarray):
                The training input samples.
            y (np.ndarray):
                The target values (class labels).
            depth (int):
                The current depth of the outer tree being built.
            outer_node_name (str):
                A unique name for the current node in the outer decision tree.
            tree (Optional[DecisionTreeClassifier], optional):
                The inner decision tree associated with this node. Defaults to None.
            group (Optional[List[int]], optional):
                The group of features associated with this node. Defaults to None.

        Returns:
            NestedTreesClassifier.Node: A node in the outer decision tree.

        """
        if depth == (self.max_outer_depth - 1) or len(self.features_group) < 2 or len(X) < self.min_outer_samples:
            return self.Node(is_leaf=True, tree=tree, node_name=outer_node_name, group=group)

        best_tree, best_split, best_group = self._find_best_tree_and_split(X, y, outer_node_name)

        if len(best_split) == 1:
            return self.Node(is_leaf=True, node_name=outer_node_name, tree=best_tree, group=best_group)

        node = self.Node(is_leaf=False, tree=best_tree, node_name=outer_node_name, group=best_group)
        self._add_children_to_node(node, best_split, depth)
        return node

    def _find_best_tree_and_split(
        self, X: np.ndarray, y: np.ndarray, outer_node_name: str
    ) -> Tuple[DecisionTreeClassifier, List[Tuple[np.ndarray, np.ndarray, int]], List[int]]:
        """Find the best inner decision tree and the associated split (i.e., the competition).

        This method evaluates all possible inner decision trees and selects the one that results
        in the lowest Gini impurity. The method can be parallelised for faster computation.

        Args:
            X (np.ndarray):
                The training input samples.
            y (np.ndarray):
                The target values (class labels).
            outer_node_name (str):
                A unique name for the current node in the outer decision tree.

        Returns:
            Tuple[DecisionTreeClassifier, List[Tuple[np.ndarray, np.ndarray, int]], List[int]]:
                A tuple containing the best inner decision tree, the associated split, and the best feature group.

        """
        min_gini = float("inf")
        best_tree = None
        subset_X = None
        best_group = None

        if self.parallel:
            tasks = [  # pragma: no cover
                _fit_inner_tree_plus_calculate_gini_ray.remote(
                    X[:, group],
                    y,
                    i,
                    outer_node_name,
                    self.max_inner_depth,
                    self.inner_estimator_hyperparameters,
                    self.save_nested_trees,
                    group,
                )
                for i, group in enumerate(self.features_group)
            ]
            results = ray.get(tasks)  # pragma: no cover
            best_tree, _, min_gini, _, subset_X, best_group = min(results, key=lambda x: x[2])  # pragma: no cover
        else:
            for i, group in enumerate(self.features_group):
                subset_X_temp = X[:, group]
                tree, _, gini = _fit_inner_tree_and_calculate_gini(
                    subset_X_temp,
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
                    subset_X = subset_X_temp
                    best_group = group

        best_split = self._create_split(X, subset_X, y, best_tree)
        return best_tree, best_split, best_group

    def _add_children_to_node(
        self, node: "NestedTreesClassifier.Node", best_split: List[Tuple[np.ndarray, np.ndarray, int]], depth: int
    ) -> None:
        """Add children to a node in the outer decision tree based on the best split.

        Args:
            node (NestedTreesClassifier.Node):
                The node in the outer decision tree to add children to.
            best_split (List[Tuple[np.ndarray, np.ndarray, int]]):
                The best split of the data, represented as a list of tuples with (X subset, y subset, leaf number).
            depth (int):
                The current depth of the node in the outer decision tree.

        """
        for i, (subset_X, subset_y, leaf_number) in enumerate(best_split):
            child_node_name = f"outer_{node.node_name}_d{depth + 1}_g{i}_l{leaf_number}"
            child_node = self._build_outer_tree(subset_X, subset_y, depth + 1, child_node_name, node.tree, node.group)
            node.children.append(child_node)
            node.children_map[leaf_number] = child_node

    def _create_split(
        self, X: np.ndarray, subset_X: np.ndarray, y: np.ndarray, tree: DecisionTreeClassifier
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """Create a split of the data based on the leaf nodes of the decision tree.

        Args:
            X (np.ndarray):
                The original feature matrix. subset_X (np.ndarray): The feature matrix for the current group.
            y (np.ndarray):
                The target labels.
            tree (DecisionTreeClassifier):
                The decision tree used for creating the split.

        Returns:
            List[Tuple[np.ndarray, np.ndarray, int]]:
                A list of tuples representing the split data, with each tuple containing:
                    * X subset corresponding to a leaf node
                    * y subset corresponding to a leaf node
                    * Leaf number

        """
        leaves = tree.apply(subset_X)
        unique_leaves = np.unique(leaves)
        return [(X[leaves == leaf], y[leaves == leaf], leaf) for leaf in unique_leaves]

    def _predict_single(self, x: np.ndarray) -> int:
        """Predict the class label for a single input sample.

        Args:
            x (np.ndarray):
                The input sample.

        Returns:
            int: The predicted class label for the input sample.

        """
        node = self.root
        leaf_subset = None

        while not node.is_leaf:
            subset_x = x[node.group]
            next_node_leaf_number = node.tree.apply([subset_x])[0]

            node = node.children_map[next_node_leaf_number]
            leaf_subset = x[node.group]

        return node.tree.predict(leaf_subset.reshape(1, -1))[0]

    def _predict_proba_single(self, x: np.ndarray) -> np.ndarray:
        """Predict the class probabilities for a single input sample.

        Args:
            x (np.ndarray): The input sample.

        Returns:
            np.ndarray: The predicted class probabilities for the input sample.

        """
        node = self.root
        leaf_subset = None

        while not node.is_leaf:
            subset_x = x[node.group]
            next_node_leaf_number = node.tree.apply([subset_x])[0]

            node = node.children_map[next_node_leaf_number]
            leaf_subset = x[node.group]

        return node.tree.predict_proba(leaf_subset.reshape(1, -1))

    def print_nested_tree(
        self,
        node: Optional["NestedTreesClassifier.Node"] = None,
        depth: int = 0,
        prefix: str = "",
        parent_name: str = "",
    ) -> None:
        """Print the nested tree structure for interpretation.

        Args:
            node (Optional[Node]): Starting node (defaults to root if None).
            depth (int): Current depth (defaults to 0).
            prefix (str): String to prepend to node names (defaults to "").
            parent_name (str): Parent node name (defaults to "").

        !!! tip "Debugging Aid"
            Use this to visualize the tree hierarchy and verify model construction.
            Careful, it could be very verbose for large trees.
        """
        if node is None:
            node = self.root

        node_name_parts = node.node_name.split("_")
        unique_node_name_parts = _remove_consecutive_duplicates(node_name_parts)
        node_name = "_".join(unique_node_name_parts)

        if parent_name:
            node_name = node_name.replace(f"{parent_name}_", "")

        if node.is_leaf:
            print(f"{prefix}* Leaf {depth}: {node_name}")
        else:
            print(f"{prefix}* Node {depth}: {node_name}")
            for child in node.children:
                self.print_nested_tree(child, depth + 1, f"{prefix}  ", node_name)
