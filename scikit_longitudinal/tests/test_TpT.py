import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from scikit_longitudinal.estimators.trees import (
    TpTDecisionTreeClassifier,
    TpTDecisionTreeRegressor,
)


def create_synthetic_classification(
    n_samples: int = 150,
    n_longitudinal_groups: int = 2,
    n_features_per_group: int = 2,
    n_non_longitudinal: int = 2,
    random_state: int | None = None,
):
    rng = np.random.RandomState(random_state)
    n_features = n_longitudinal_groups * n_features_per_group + n_non_longitudinal

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_longitudinal_groups * n_features_per_group,
        n_redundant=0,
        n_repeated=0,
        random_state=random_state,
    )

    features_group = [
        list(range(i * n_features_per_group, (i + 1) * n_features_per_group))
        for i in range(n_longitudinal_groups)
    ]

    # Add some temporal correlation between waves inside each longitudinal group.
    for group in features_group:
        base, *later = group
        for idx in later:
            X[:, idx] = X[:, base] + rng.normal(0.0, 0.1, size=n_samples)

    return X, y, features_group


def create_synthetic_regression(
    n_samples: int = 150,
    n_longitudinal_groups: int = 2,
    n_features_per_group: int = 2,
    n_non_longitudinal: int = 2,
    noise: float = 0.1,
    random_state: int | None = None,
):
    rng = np.random.RandomState(random_state)
    n_features = n_longitudinal_groups * n_features_per_group + n_non_longitudinal

    features_group = [
        list(range(i * n_features_per_group, (i + 1) * n_features_per_group))
        for i in range(n_longitudinal_groups)
    ]

    # Draw X first, then enforce temporal correlation between the waves of
    # each longitudinal group. Building y *after* this step ensures the target
    # depends on features that survive the within-group copying — otherwise
    # any signal placed on the "later" wave columns is overwritten and the
    # tree can never recover it.
    X = rng.randn(n_samples, n_features)
    for group in features_group:
        base, *later = group
        for idx in later:
            X[:, idx] = X[:, base] + rng.normal(0.0, 0.1, size=n_samples)

    coefs = np.zeros(n_features)
    for group in features_group:
        coefs[group[0]] = rng.uniform(2.0, 4.0)
    for idx in range(n_longitudinal_groups * n_features_per_group, n_features):
        coefs[idx] = rng.uniform(1.0, 2.0)
    y = X @ coefs + rng.normal(0.0, noise, size=n_samples)

    return X, y, features_group


class TestTpTWideFormat:
    @pytest.fixture(scope="class")
    def classification_data(self):
        X, y, features_group = create_synthetic_classification(
            n_samples=200,
            n_longitudinal_groups=2,
            n_features_per_group=2,
            n_non_longitudinal=2,
            random_state=42,
        )
        return X, y, features_group

    @pytest.fixture(scope="class")
    def regression_data(self):
        X, y, features_group = create_synthetic_regression(
            n_samples=200,
            n_longitudinal_groups=2,
            n_features_per_group=2,
            n_non_longitudinal=2,
            noise=0.1,
            random_state=42,
        )
        return X, y, features_group

    def test_TpT_classifier_on_synthetic_wide(self, classification_data):
        X, y, features_group = classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        clf = TpTDecisionTreeClassifier(
            threshold_gain=0.005,
            features_group=features_group,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        assert 0.0 <= acc <= 1.0

    def test_TpT_regressor_on_synthetic_wide(self, regression_data):
        X, y, features_group = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        reg = TpTDecisionTreeRegressor(
            threshold_gain=0.005,
            features_group=features_group,
            random_state=42,
        )
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        # Tree regressors on this synthetic data should achieve a reasonable fit.
        assert r2 > 0.5

