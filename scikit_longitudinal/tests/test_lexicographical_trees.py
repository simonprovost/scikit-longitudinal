import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from scikit_longitudinal.estimators.trees import LexicoDecisionTreeClassifier, LexicoRFClassifier


def create_synthetic_data(
    n_samples=100, n_longitudinal_groups=2, n_features_per_group=2, n_non_longitudinal=2, random_state=None
):
    np.random.seed(random_state)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_longitudinal_groups * n_features_per_group + n_non_longitudinal,
        random_state=random_state,
    )

    features_group = [
        list(range(i * n_features_per_group, (i + 1) * n_features_per_group)) for i in range(n_longitudinal_groups)
    ]

    for i in range(n_longitudinal_groups - 1):
        X[:, features_group[i + 1]] = X[:, features_group[i]] + np.random.normal(
            0, 0.1, size=(n_samples, n_features_per_group)
        )

    return X, y, features_group


class TestLexico:
    @pytest.fixture(scope="class")
    def load_iris_data(self):
        return load_iris()

    @pytest.fixture(scope="class")
    def train_test_data_iris(self, load_iris_data):
        return train_test_split(load_iris_data.data, load_iris_data.target, test_size=0.3, random_state=42)

    @pytest.fixture(scope="class")
    def synthetic_data(self):
        X, y, features_group = create_synthetic_data(
            n_samples=150, n_longitudinal_groups=2, n_features_per_group=2, n_non_longitudinal=2, random_state=42
        )
        return X, y, features_group

    @pytest.fixture(scope="class")
    def train_test_data_synthetic(self, synthetic_data):
        X, y, _ = synthetic_data
        return train_test_split(X, y, test_size=0.3, random_state=42)

    @pytest.mark.parametrize(
        "threshold_gain, features_group, n_estimators, random_state",
        [
            (0.5, [[0, 1], [2, 3]], 10, 42),
            (0.1, [[0, 1], [2, 3]], 100, 123),
            (0.25, [[0, 1], [2, 3]], 500, 321),
        ],
    )
    def test_lexico_RF_iris(self, train_test_data_iris, threshold_gain, features_group, n_estimators, random_state):
        X_train, X_test, y_train, y_test = train_test_data_iris
        clf = LexicoRFClassifier(
            n_estimators=n_estimators,
            threshold_gain=threshold_gain,
            features_group=features_group,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        assert 0 <= accuracy <= 1

    @pytest.mark.parametrize(
        "threshold_gain, features_group, random_state",
        [
            (0.5, [[0, 1], [2, 3]], 42),
            (0.1, [[0, 1], [2, 3]], 123),
            (0.25, [[0, 1], [2, 3]], 321),
        ],
    )
    def test_lexico_decision_tree_iris(self, train_test_data_iris, threshold_gain, features_group, random_state):
        X_train, X_test, y_train, y_test = train_test_data_iris
        clf = LexicoDecisionTreeClassifier(
            threshold_gain=threshold_gain, features_group=features_group, random_state=random_state
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        assert 0 <= accuracy <= 1

    @pytest.mark.parametrize(
        "threshold_gain, n_estimators, random_state",
        [
            (0.5, 1, 42),
            (0.5, 10, 42),
            (0.5, 100, 42),
            (0.5, 500, 42),
            (0.1, 1, 123),
            (0.1, 10, 123),
            (0.1, 100, 123),
            (0.1, 500, 123),
            (0.25, 1, 321),
            (0.25, 10, 321),
            (0.25, 100, 321),
        ],
    )
    def test_lexico_RF_synthetic(
        self, train_test_data_synthetic, synthetic_data, threshold_gain, n_estimators, random_state
    ):
        X_train, X_test, y_train, y_test = train_test_data_synthetic
        _, _, features_group = synthetic_data
        clf = LexicoRFClassifier(
            n_estimators=n_estimators,
            threshold_gain=threshold_gain,
            features_group=features_group,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        assert 0 <= accuracy <= 1

    @pytest.mark.parametrize(
        "threshold_gain, random_state",
        [
            (0.5, 42),
            (0.1, 123),
            (0.25, 321),
            (0.05, 421),
        ],
    )
    def test_lexico_decision_tree_synthetic(
        self, train_test_data_synthetic, synthetic_data, threshold_gain, random_state
    ):
        X_train, X_test, y_train, y_test = train_test_data_synthetic
        _, _, features_group = synthetic_data
        clf = LexicoDecisionTreeClassifier(
            threshold_gain=threshold_gain, features_group=features_group, random_state=random_state
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        assert 0 <= accuracy <= 1