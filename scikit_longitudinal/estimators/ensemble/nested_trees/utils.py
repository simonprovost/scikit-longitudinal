from typing import Any, Dict, List, Tuple

import numpy as np
import ray
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# pylint: disable=R0913


@ray.remote
def _fit_inner_tree_plus_calculate_gini_ray(
    subset_X: np.ndarray,
    y: np.ndarray,
    group_index: int,
    outer_node_name: str,
    max_inner_depth: int,
    inner_estimator_hyperparameters: Dict[str, Any],
    save_nested_trees: bool,
    group: List[int],
) -> Tuple[DecisionTreeClassifier, Any, float, np.ndarray, List[int]]:
    """
    Copy of _fit_inner_tree_plus_calculate_gini to be used with Ray parallelization.
    """
    tree, y_pred, gini = _fit_inner_tree_and_calculate_gini(
        subset_X, y, group_index, outer_node_name, max_inner_depth, inner_estimator_hyperparameters, save_nested_trees
    )
    return tree, outer_node_name, gini, y_pred, subset_X, group


def _fit_inner_tree_and_calculate_gini(
    subset_X: np.ndarray,
    y: np.ndarray,
    group_index: int,
    outer_node_name: str,
    max_inner_depth: int,
    inner_estimator_hyperparameters: Dict[str, Any],
    save_nested_trees: bool,
) -> Tuple[DecisionTreeClassifier, np.ndarray, float]:
    """Copy of _fit_inner_tree_plus_calculate_gini to be used with Ray parallelization.

    Args:
        subset_X (np.ndarray):
            The training input samples for a specific group of features.
        y (np.ndarray):
            The target values (class labels).
        group_index (int):
            The index of the current group of features.
        outer_node_name (str):
            A unique name for the current node in the outer decision tree.
        max_inner_depth (int):
            The maximum depth for the inner decision tree.
        inner_estimator_hyperparameters (Dict[str, Any]):
            A dictionary of hyperparameters for the inner decision tree.
        save_nested_trees (bool):
            If True, save the inner trees as images.

    Returns:
        Tuple[DecisionTreeClassifier, Any, float, np.ndarray, List[int]]: A tuple containing the fitted inner decision
        tree, the outer node name, the Gini impurity, the predicted labels, and the current group of features.

    """
    tree = DecisionTreeClassifier(max_depth=max_inner_depth, **inner_estimator_hyperparameters)
    tree.fit(subset_X, y)
    if save_nested_trees:  # pragma: no cover
        _save_inner_tree(tree, f"inner_tree_{outer_node_name}_group_{group_index}.png")
    y_pred = tree.predict(subset_X)
    gini = _calculate_gini(y, y_pred)
    return tree, y_pred, gini


def _calculate_gini(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Gini impurity of the predictions.

    Args:
        y (np.ndarray):
            The true labels.
        y_pred (np.ndarray):
            The predicted labels.

    Returns:
        float: The Gini impurity of the predictions.

    Formula inspired from the original implementation of the Nested Trees algorithm:
    https://github.com/NestedTrees/NestedTrees/blob/main/src/ModelEvaluator.java#L105

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


def _save_inner_tree(tree: DecisionTreeClassifier, filename: str) -> None:  # pragma: no cover
    """Save a visual representation of the given decision tree as an image file.

    Args:
        tree (DecisionTreeClassifier):
            The decision tree to visualize.
        filename (str):
            The name of the file to save the image as.

    """
    plt.figure(figsize=(10, 10))
    plot_tree(tree, filled=True)
    plt.savefig(filename)
    plt.close()


def _remove_consecutive_duplicates(values: List[str]) -> List[str]:
    """Remove consecutive duplicates in a list of strings.

    Args:
        values (List[str]):
            The list of strings to process.

    Returns:
        List[str]:
            The list of strings with consecutive duplicates removed.

    Examples were taken from the following string node printed from the print_nested_tree method:

    "outer_node_outer_node_..." -> "outer_node_..."

    """
    result = []
    prev = None
    for item in values:
        if item != prev:
            result.append(item)
        prev = item
    return result
