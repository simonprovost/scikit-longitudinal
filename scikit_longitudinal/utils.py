import os  # pragma: no cover
from typing import List, Union  # pragma: no cover

import graphviz  # pragma: no cover
import matplotlib.pyplot as plt  # pragma: no cover
import numpy as np  # pragma: no cover
from sklearn.ensemble import RandomForestClassifier  # pragma: no cover
from sklearn.tree import DecisionTreeClassifier, export_graphviz  # pragma: no cover

from scikit_longitudinal.estimators.tree import LexicoDecisionTree, LexicoRF  # pragma: no cover


# pylint: disable=W9016
def save_feature_importances(
    clf, feature_names: List[str], top_n: int = 10, output_path: str = "output/feature_importances.png"
) -> None:  # pragma: no cover
    """Saves the top_n features with the greatest average impurity decrease (AID)

    Plot using the feature_importances_ attribute from a trained classifier.

    Args:
        clf:
            A trained classifier with a feature_importances_ attribute.
        feature_names (List[str]):
            List of feature names.
        top_n (int, optional):
            Number of top features to plot. Defaults to 10.
        output_path (str, optional):
            The path to save the plot image. Defaults to "feature_importances.png".ts to
            "feature_importances.png".ts to "feature_importances.png".ts to "feature_importances.png".ts
            to "feature_importances.png".

    """
    # Get the feature importances
    importances = clf.feature_importances_

    # Get the indices of the features sorted by importance
    sorted_indices = np.argsort(importances)[::-1]

    # Get the top_n feature names and their importances
    top_feature_names = [feature_names[i] for i in sorted_indices[:top_n]]
    top_importances = importances[sorted_indices[:top_n]]

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.bar(top_feature_names, top_importances)
    plt.xlabel("Feature")
    plt.ylabel("Average Impurity Decrease")
    plt.title(f"Top {top_n} Features with Greatest Average Impurity Decrease (AID)")

    # Save the plot
    plt.savefig(output_path)
    plt.close()


def save_trees_graphviz(
    model: Union[DecisionTreeClassifier, RandomForestClassifier],
    feature_names: List[str],
    class_names: List[str],
    output_dir: str = "output",
) -> None:  # pragma: no cover
    """Save a list of graphviz.Source figure for decision tree(s) from a classifier.

    Args:
        model (Union[DecisionTreeClassifier, RandomForestClassifier]):
            A trained decision tree classifier or a random forest classifier.
        feature_names (List[str]):
            List of feature names.
        class_names (List[str]):
            List of class names.
        output_dir (str, optional):
            Output directory. Defaults to 'output'.

    """
    if isinstance(model, (RandomForestClassifier, LexicoRF)):
        trees = model.estimators_
    elif isinstance(model, (DecisionTreeClassifier, LexicoDecisionTree)):
        trees = [model]
    else:
        raise ValueError("Invalid model. Expected DecisionTreeClassifier or RandomForestClassifier.")

    for i, tree in enumerate(trees):
        dot_data = export_graphviz(
            tree,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True,
            out_file=None,
        )
        dot_data_with_title = "digraph Tree {\n" + f'label="Decision Tree {i + 1}";\n' + dot_data.split("{", 1)[1]

        graph = graphviz.Source(dot_data_with_title)
        file_path = os.path.join(output_dir, f"decision_tree_{i + 1}.gv")
        graph.render(file_path, cleanup=True)
