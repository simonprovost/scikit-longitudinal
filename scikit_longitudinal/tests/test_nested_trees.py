# pylint: disable=W0621,W0613,W0212
import io
import sys

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from scikit_longitudinal.estimators.ensemble import NestedTreesClassifier


@pytest.fixture
def dummy_data():
    n_samples = 100
    n_features = 6
    X = np.linspace(0, 1, n_samples * n_features).reshape(n_samples, n_features)
    y = np.array([0 if i % 2 == 0 else 1 for i in range(n_samples)])
    return X, y


@pytest.fixture
def features_group():
    return [[0, 1, 2], [2, 3, 4]]


class TestNestedTreesClassifier:
    def test_init_and_default_parameters(self, features_group):
        classifier = NestedTreesClassifier(features_group)
        assert classifier.features_group == features_group
        assert classifier.max_outer_depth == 3
        assert classifier.max_inner_depth == 2
        assert classifier.min_outer_samples == 5
        assert classifier.inner_estimator_hyperparameters is None
        assert classifier.root is None

    def test_fit_and_predict(self, dummy_data, features_group):
        X, y = dummy_data
        nested_trees = NestedTreesClassifier(features_group)
        nested_trees.fit(X, y)

        y_pred = nested_trees.predict(X)
        accuracy = (y_pred == y).mean()
        assert accuracy >= 0

    def test_single_sample_prediction(self, dummy_data, features_group):
        X, y = dummy_data
        nested_trees = NestedTreesClassifier(features_group)
        nested_trees.fit(X, y)

        x_single = X[0, :]
        y_single_pred = nested_trees._predict_single(x_single)
        assert y_single_pred in np.unique(y)

    def test_hyperparameters_tuning(self, dummy_data, features_group):
        X, y = dummy_data
        nested_trees = NestedTreesClassifier(
            features_group,
            max_outer_depth=4,
            max_inner_depth=3,
            min_outer_samples=2,
            inner_estimator_hyperparameters={"min_samples_split": 10, "max_features": "sqrt"},
        )
        nested_trees.fit(X, y)

        y_pred = nested_trees.predict(X)
        accuracy = (y_pred == y).mean()
        assert accuracy >= 0
        assert nested_trees.max_outer_depth == 4
        assert nested_trees.max_inner_depth == 3
        assert nested_trees.min_outer_samples == 2
        assert nested_trees.inner_estimator_hyperparameters == {"min_samples_split": 10, "max_features": "sqrt"}

    def test_tree_structure_as_string(self, dummy_data, features_group):
        X, y = dummy_data
        nested_trees = NestedTreesClassifier(features_group)
        nested_trees.fit(X, y)

        tree_output = capture_print_nested_tree_output(nested_trees)
        expected_output = get_expected_tree_output()

        tree_output_no_spaces = "".join(c for c in tree_output if c.isalnum())
        expected_output_no_spaces = "".join(c for c in expected_output if c.isalnum())

        assert tree_output_no_spaces == expected_output_no_spaces

    def test_find_best_tree_and_split(self, dummy_data, features_group):
        X, y = dummy_data
        classifier = NestedTreesClassifier(features_group)
        classifier.fit(X, y)
        best_tree, best_split, best_group = classifier._find_best_tree_and_split(X, y, "test_node")

        assert isinstance(best_tree, DecisionTreeClassifier)

        assert len(best_split) > 0

        assert len(best_group) in [len(features_group[0]), len(features_group[1])]

        assert np.array_equal(best_group, features_group[0]) or np.array_equal(best_group, features_group[1])

    def test_node_init_and_str_method(self, dummy_data, features_group):
        X, y = dummy_data
        tree = DecisionTreeClassifier()
        tree.fit(X, y)
        node = NestedTreesClassifier.Node(is_leaf=False, tree=tree, node_name="dummy_node", group=features_group[0])
        assert str(node) == "dummy_node"

    @pytest.mark.parametrize(
        "max_outer_depth, max_inner_depth, min_outer_samples, features_group",
        [
            (0, 2, 5, [[0, 1, 2], [2, 3, 4]]),
            (3, 0, 5, [[0, 1, 2], [2, 3, 4]]),
            (3, 2, 0, [[0, 1, 2], [2, 3, 4]]),
        ],
    )
    def test_invalid_init_params(self, max_outer_depth, max_inner_depth, min_outer_samples, features_group):
        with pytest.raises(ValueError):
            NestedTreesClassifier(
                features_group,
                None,
                max_outer_depth=max_outer_depth,
                max_inner_depth=max_inner_depth,
                min_outer_samples=min_outer_samples,
            )

    @pytest.mark.parametrize(
        "is_leaf, tree, node_name, group",
        [
            (False, None, "dummy_node", [0, 1, 2]),
            (False, DecisionTreeClassifier(), "", [0, 1, 2]),
        ],
    )
    def test_node_invalid_init_params(self, is_leaf, tree, node_name, group):
        with pytest.raises(ValueError):
            NestedTreesClassifier.Node(is_leaf=is_leaf, tree=tree, node_name=node_name, group=group)

    def test_nested_trees_classifier_fit_invalid_features_group(self, dummy_data, features_group):
        X, y = dummy_data
        fake = [[0, 1, 2]]
        classifier = NestedTreesClassifier(features_group)
        classifier.features_group = fake

        with pytest.raises(ValueError):
            classifier.fit(X, y)

    def test_nested_trees_classifier_predict_before_fit(self, dummy_data):
        X, y = dummy_data
        features_group = [[0, 1, 2], [3, 4, 5]]
        classifier = NestedTreesClassifier(features_group)

        with pytest.raises(ValueError):
            classifier.predict(X)

    @pytest.mark.parametrize(
        "X, expected",
        [
            (np.array([[0, 1, 2, 3, 4, 5]]), np.array([0])),
        ],
    )
    def test_predict_output_shape(self, X, expected, dummy_data, features_group):
        X_train, y_train = dummy_data
        classifier = NestedTreesClassifier(features_group)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X)
        y_pred_proba = classifier.predict_proba(X)
        assert y_pred is not None
        assert y_pred.shape == expected.shape
        assert y_pred_proba is not None
        assert y_pred_proba.shape == (1, 2)

    @pytest.mark.parametrize(
        "X",
        [
            (np.array([[0, 1, 2, 3, 4, 5]])),
        ],
    )
    def test_predict_with_no_root(self, X, dummy_data, features_group):
        X_train, y_train = dummy_data
        classifier = NestedTreesClassifier(features_group)
        classifier.fit(X_train, y_train)
        classifier.root = None
        with pytest.raises(ValueError, match="The classifier must be fitted before making predictions."):
            classifier.predict(X)
        with pytest.raises(ValueError, match="The classifier must be fitted before making predictions."):
            classifier.predict_proba(X)


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
          * Leaf 1: d1_g0_l1
          * Leaf 1: d1_g1_l3
        * Node 1: d1_g2_l4
          * Leaf 2: outer_root_d2_g0_l1
          * Leaf 2: outer_root_d2_g1_l3
          * Leaf 2: outer_root_d2_g2_l4
    """
