import numpy as np
import pytest
from sklearn.datasets import make_classification

from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_deep_forest import (
    LexicoDeepForestClassifier,
    LongitudinalClassifierType,
    LongitudinalEstimatorConfig,
)


@pytest.fixture
def synthetic_data():
    n_samples = 100
    n_features_per_group = 2
    n_longitudinal_groups = 2
    n_non_longitudinal = 2
    random_state = 42

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

    return X, y, features_group, n_non_longitudinal


@pytest.fixture
def uninitialized_classifier():
    return LexicoDeepForestClassifier(
        features_group=[[0, 1], [2, 3]],
        longitudinal_base_estimators=[
            LongitudinalEstimatorConfig(classifier_type=LongitudinalClassifierType.LEXICO_RF)
        ],
    )


class TestDeepForest:
    def test_deep_forests_longitudinal_classifier_initialization(self, synthetic_data):
        _, _, features_group, _ = synthetic_data
        classifier = LexicoDeepForestClassifier(
            features_group=features_group,
            longitudinal_base_estimators=[
                LongitudinalEstimatorConfig(classifier_type=LongitudinalClassifierType.LEXICO_RF)
            ],
        )
        assert classifier is not None

    def test_ensure_valid_state_decorator_without_fitting(self, synthetic_data):
        X, _, features_group, _ = synthetic_data
        classifier = LexicoDeepForestClassifier(
            features_group=features_group,
            longitudinal_base_estimators=[
                LongitudinalEstimatorConfig(classifier_type=LongitudinalClassifierType.LEXICO_RF)
            ],
        )

        assert classifier is not None

        with pytest.raises(ValueError):
            classifier.predict(X)
        with pytest.raises(ValueError):
            classifier.predict_proba(X)

    def test_longitudinal_estimator_config_initialization(self):
        config = LongitudinalEstimatorConfig(classifier_type=LongitudinalClassifierType.LEXICO_RF)
        assert config.classifier_type == LongitudinalClassifierType.LEXICO_RF

    def test_example_1_from_docstring(self, synthetic_data):
        X, y, features_group, n_non_longitudinal = synthetic_data
        non_longitudinal_features = list(range(-n_non_longitudinal, 0))
        lexico_rf_config = LongitudinalEstimatorConfig(
            classifier_type=LongitudinalClassifierType.LEXICO_RF,
            count=2,
            hyperparameters={"max_depth": 2, "n_estimators": 2},
        )
        clf = LexicoDeepForestClassifier(
            features_group=features_group,
            non_longitudinal_features=non_longitudinal_features,
            longitudinal_base_estimators=[lexico_rf_config],
            random_state=42,
        )
        clf.fit(X, y)
        predictions = clf.predict(X)
        prediction_proba = clf.predict_proba(X)
        assert predictions is not None
        assert prediction_proba is not None
        assert len(predictions) == len(y)

    def test_example_2_from_docstring(self, synthetic_data):
        X, y, features_group, n_non_longitudinal = synthetic_data
        non_longitudinal_features = list(range(-n_non_longitudinal, 0))
        lexico_rf_config = LongitudinalEstimatorConfig(
            classifier_type=LongitudinalClassifierType.LEXICO_RF,
            count=2,
            hyperparameters={"max_depth": 2, "n_estimators": 2},
        )
        complete_random_lexico_rf = LongitudinalEstimatorConfig(
            classifier_type=LongitudinalClassifierType.COMPLETE_RANDOM_LEXICO_RF,
            count=2,
            hyperparameters={"max_depth": 3, "n_estimators": 5},
        )
        clf = LexicoDeepForestClassifier(
            features_group=features_group,
            non_longitudinal_features=non_longitudinal_features,
            longitudinal_base_estimators=[lexico_rf_config, complete_random_lexico_rf],
            random_state=42,
        )
        clf.fit(X, y)
        predictions = clf.predict(X)
        prediction_proba = clf.predict_proba(X)
        assert predictions is not None
        assert prediction_proba is not None
        assert len(predictions) == len(y)

    def test_example_3_from_docstring(self, synthetic_data):
        X, y, features_group, n_non_longitudinal = synthetic_data
        non_longitudinal_features = list(range(-n_non_longitudinal, 0))
        lexico_rf_config = LongitudinalEstimatorConfig(
            classifier_type=LongitudinalClassifierType.LEXICO_RF,
            count=2,
            hyperparameters={"max_depth": 2, "n_estimators": 2},
        )
        clf = LexicoDeepForestClassifier(
            features_group=features_group,
            non_longitudinal_features=non_longitudinal_features,
            longitudinal_base_estimators=[lexico_rf_config],
            diversity_estimators=False,
            random_state=42,
        )
        assert clf.diversity_estimators is False
        clf.fit(X, y)
        predictions = clf.predict(X)
        prediction_proba = clf.predict_proba(X)
        assert predictions is not None
        assert prediction_proba is not None
        assert len(predictions) == len(y)

    def test_invalid_features_group(self, synthetic_data, uninitialized_classifier):
        X, y, _, _ = synthetic_data
        uninitialized_classifier.features_group = None
        with pytest.raises(ValueError) as e:
            uninitialized_classifier.fit(X, y)
        assert str(e.value) == "features_group must contain more than one feature group."

    def test_missing_longitudinal_base_estimators(self, synthetic_data):
        X, y, features_group, _ = synthetic_data
        classifier = LexicoDeepForestClassifier(features_group=features_group)
        with pytest.raises(ValueError) as e:
            classifier.fit(X, y)
        assert str(e.value) == "longitudinal_base_estimators must be provided."

    def test_missing_diversity_estimators(self, synthetic_data):
        X, y, features_group, _ = synthetic_data
        classifier = LexicoDeepForestClassifier(
            features_group=features_group,
            longitudinal_base_estimators=[
                LongitudinalEstimatorConfig(classifier_type=LongitudinalClassifierType.LEXICO_RF)
            ],
            diversity_estimators=None,
        )
        with pytest.raises(ValueError) as e:
            classifier.fit(X, y)
        assert str(e.value) == "diversity_estimators must be provided. True or False."
