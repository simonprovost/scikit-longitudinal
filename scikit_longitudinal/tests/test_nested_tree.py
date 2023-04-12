# pylint: disable=W0621,W0613,W0212
import io
import sys

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from scikit_longitudinal.estimators.tree import NestedTreesClassifier


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n_samples = 100
    n_features = 6
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(2, size=n_samples)
    return X, y


@pytest.fixture
def grouping_structure():
    return [[0, 1, 2], [2, 3, 4]]


class TestNestedTreesClassifier:
    def test_init(self, grouping_structure):
        classifier = NestedTreesClassifier(grouping_structure)
        assert classifier.group_features == grouping_structure
        assert classifier.max_outer_depth == 3
        assert classifier.max_inner_depth == 2
        assert classifier.min_outer_samples == 5
        assert classifier.inner_estimator_hyperparameters == {}
        assert classifier.root is None

    @pytest.mark.parametrize(
        "X, expected",
        [
            (np.array([[0, 1, 2, 3, 4, 5]]), np.array([0])),
        ],
    )
    def test_predict(self, X, expected, dummy_data, grouping_structure):
        X_train, y_train = dummy_data
        classifier = NestedTreesClassifier(grouping_structure)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X)
        assert np.array_equal(y_pred, expected)

    @pytest.mark.parametrize("X, y", [(np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]), np.array([0, 1]))])
    def test_accuracy(self, X, y, dummy_data, grouping_structure):
        X_train, y_train = dummy_data
        classifier = NestedTreesClassifier(grouping_structure)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_train)
        accuracy = (y_pred == y_train).mean()
        assert accuracy >= 0.45

    def test_find_best_tree_and_split(self, dummy_data, grouping_structure):
        X, y = dummy_data
        classifier = NestedTreesClassifier(grouping_structure)
        classifier.fit(X, y)
        best_tree, best_split = classifier._find_best_tree_and_split(X, y, "test_node")
        assert isinstance(best_tree, DecisionTreeClassifier)
        assert len(best_split) > 0

    def test_precise_accuracy(self, dummy_data, grouping_structure):
        X, y = dummy_data
        nested_trees = NestedTreesClassifier(grouping_structure)
        nested_trees.fit(X, y)

        y_pred = nested_trees.predict(X)
        accuracy = (y_pred == y).mean()
        assert accuracy == 0.46

    def test_tree_structure_as_string(self, dummy_data, grouping_structure):
        X, y = dummy_data
        nested_trees = NestedTreesClassifier(grouping_structure)
        nested_trees.fit(X, y)

        tree_output = capture_print_nested_tree_output(nested_trees)
        expected_output = get_expected_tree_output()

        tree_output_no_spaces = "".join(c for c in tree_output if c.isalnum())
        expected_output_no_spaces = "".join(c for c in expected_output if c.isalnum())

        assert tree_output_no_spaces == expected_output_no_spaces


def capture_print_nested_tree_output(classifier):
    original_stdout = sys.stdout
    sys.stdout = captured_stdout = io.StringIO()

    try:
        classifier.print_nested_tree()
    finally:
        sys.stdout = original_stdout

    return captured_stdout.getvalue()


def get_expected_tree_output():
    return """
        * Node 0: outer_root
          * Node 1: d1_g0_l2
            * Leaf 2: outer_root_d2_g0_l2
            * Leaf 2: outer_root_d2_g1_l3
          * Node 1: d1_g1_l3
            * Leaf 2: outer_root_d2_g0_l2
            * Leaf 2: outer_root_d2_g1_l3
            * Leaf 2: outer_root_d2_g2_l5
            * Leaf 2: outer_root_d2_g3_l6
          * Leaf 1: d1_g2_l5
          * Node 1: d1_g3_l6
            * Leaf 2: outer_root_d2_g0_l1
            * Leaf 2: outer_root_d2_g1_l3
    """
