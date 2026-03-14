import warnings

import numpy as np
import pytest
from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.estimators.ensemble.longitudinal_stacking.longitudinal_stacking import (
    LongitudinalStackingClassifier,
)


class FeatureCountClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, expected_n_features):
        self.expected_n_features = expected_n_features

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if X.shape[1] != self.expected_n_features:
            raise ValueError(
                f"Expected {self.expected_n_features} features, got {X.shape[1]}."
            )

        self.classes_ = unique_labels(y)
        class_counts = np.bincount(y)
        self.majority_class_ = np.argmax(class_counts)
        return self

    def predict(self, X):
        check_is_fitted(self, ["classes_", "majority_class_"])
        X = check_array(X)
        if X.shape[1] != self.expected_n_features:
            raise ValueError(
                f"Expected {self.expected_n_features} features, got {X.shape[1]}."
            )
        return np.full(X.shape[0], self.majority_class_, dtype=int)

    def predict_proba(self, X):
        check_is_fitted(self, ["classes_", "majority_class_"])
        X = check_array(X)
        if X.shape[1] != self.expected_n_features:
            raise ValueError(
                f"Expected {self.expected_n_features} features, got {X.shape[1]}."
            )

        probabilities = np.zeros((X.shape[0], len(self.classes_)))
        probabilities[:, np.where(self.classes_ == self.majority_class_)[0][0]] = 1.0
        return probabilities


class NoPredictProbaClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.majority_class_ = self.classes_[0]
        return self

    def predict(self, X):
        check_is_fitted(self, ["classes_", "majority_class_"])
        X = check_array(X)
        return np.full(X.shape[0], self.majority_class_, dtype=self.classes_.dtype)


class TestLongitudinalStackingClassifier:
    @pytest.fixture
    def data(self):
        longitudinal_data = LongitudinalDataset(
            "scikit_longitudinal/tests/dummy_data/dummy_data_3.csv"
        )
        longitudinal_data.load_data_target_train_test_split(
            target_column="target_w2",
            remove_target_waves=True,
            target_wave_prefix="target_",
        )
        longitudinal_data.sety_train(
            [0 if y == " A" else 1 for y in longitudinal_data.y_train]
        )
        longitudinal_data.sety_test(
            [0 if y == " A" else 1 for y in longitudinal_data.y_test]
        )
        longitudinal_data.setup_features_group(input_data="elsa")

        y_train = longitudinal_data.y_train.copy()
        for i in range(len(y_train) // 2):
            y_train[i] += 1

        x_test = longitudinal_data.X_test.copy()
        for i in range(len(x_test) // 2):
            x_test.iloc[i, 0] += 1

        longitudinal_data.sety_train(y_train)
        longitudinal_data.setX_test(x_test)

        return longitudinal_data

    @pytest.fixture
    def valid_estimators(self, data):
        estimator = RandomForestClassifier(max_depth=10, random_state=0)
        return [
            (f"estimator_{i}", estimator.fit(data.X_train, data.y_train))
            for i in range(3)
        ]

    @pytest.fixture
    def valid_meta_learner(self):
        return LogisticRegression()

    @pytest.fixture
    def multiclass_wave_data(self):
        X, y = make_classification(
            n_samples=180,
            n_features=6,
            n_informative=6,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            class_sep=1.5,
            random_state=13,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=13
        )
        features_group = [[0, 1, 2], [3, 4, 5]]
        return X_train, X_test, y_train, y_test, features_group

    def test_fit_no_estimators_raises_value_error(self, data):
        with pytest.raises(ValueError):
            LongitudinalStackingClassifier(estimators=[]).fit(
                data.X_train, data.y_train
            )

    def test_fit_invalid_meta_learner_raises_value_error(self, data, valid_estimators):
        with pytest.raises(ValueError):
            LongitudinalStackingClassifier(
                estimators=valid_estimators, meta_learner="invalid"
            ).fit(data.X_train, data.y_train)

    def test_fit_with_valid_input_initializes_clf_ensemble(
        self, data, valid_estimators, valid_meta_learner
    ):
        classifier = LongitudinalStackingClassifier(
            estimators=valid_estimators, meta_learner=valid_meta_learner
        )
        classifier._fit(data.X_train, data.y_train)
        assert classifier.clf_ensemble is not None

    def test_predict_on_unfitted_model_raises_not_fitted_error(
        self, data, valid_estimators, valid_meta_learner
    ):
        classifier = LongitudinalStackingClassifier(
            estimators=valid_estimators, meta_learner=valid_meta_learner
        )
        with pytest.raises(NotFittedError):
            classifier._predict(data.X_test)

    def test_predict_on_fitted_model_returns_predictions(
        self, data, valid_estimators, valid_meta_learner
    ):
        classifier = LongitudinalStackingClassifier(
            estimators=valid_estimators, meta_learner=valid_meta_learner
        )
        classifier._fit(data.X_train, data.y_train)
        predictions = classifier._predict(data.X_test)
        assert isinstance(predictions, ndarray)

    def test_predict_proba_on_unfitted_model_raises_not_fitted_error(
        self, data, valid_estimators, valid_meta_learner
    ):
        classifier = LongitudinalStackingClassifier(
            estimators=valid_estimators, meta_learner=valid_meta_learner
        )
        with pytest.raises(NotFittedError):
            classifier._predict_proba(data.X_test)

    def test_predict_proba_on_fitted_model_returns_probabilities(
        self, data, valid_estimators, valid_meta_learner
    ):
        classifier = LongitudinalStackingClassifier(
            estimators=valid_estimators, meta_learner=valid_meta_learner
        )
        classifier._fit(data.X_train, data.y_train)
        probabilities = classifier._predict_proba(data.X_test)
        assert isinstance(probabilities, ndarray)

    def test_multiclass_stacking_supports_classes_predict_and_predict_proba(
        self, multiclass_wave_data
    ):
        X_train, X_test, y_train, _, features_group = multiclass_wave_data
        estimators = []
        for wave_index, estimator_name in enumerate(
            ["wave-old", "wave-mid", "wave-new"]
        ):
            feature_indices = [group[wave_index] for group in features_group]
            estimator = RandomForestClassifier(
                max_depth=5, random_state=wave_index
            ).fit(X_train[:, feature_indices], y_train)
            estimators.append((estimator_name, estimator))

        def extract_wave(wave, extract_indices=False):
            feature_indices = [group[wave] for group in features_group]
            if extract_indices:
                return None, None, feature_indices
            return None, None

        classifier = LongitudinalStackingClassifier(
            estimators=estimators,
            meta_learner=LogisticRegression(max_iter=200),
            extract_wave=extract_wave,
        )
        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)

        assert np.array_equal(classifier.classes_, np.unique(y_train))
        assert probabilities.shape == (len(X_test), len(classifier.classes_))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.all(np.isin(predictions, classifier.classes_))
        assert classifier.clf_ensemble.stack_method_ == ["predict_proba"] * len(
            estimators
        )

    def test_extract_wave_is_used_inside_stacking(self, multiclass_wave_data):
        X_train, X_test, y_train, _, features_group = multiclass_wave_data

        def extract_wave(wave, extract_indices=False):
            feature_indices = [group[wave] for group in features_group]
            if extract_indices:
                return None, None, feature_indices
            return None, None

        classifier = LongitudinalStackingClassifier(
            estimators=[
                ("alpha", FeatureCountClassifier(expected_n_features=2)),
                ("beta", FeatureCountClassifier(expected_n_features=2)),
                ("gamma", FeatureCountClassifier(expected_n_features=2)),
            ],
            meta_learner=LogisticRegression(max_iter=200),
            extract_wave=extract_wave,
        )

        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        assert predictions.shape == (len(X_test),)
        assert classifier.clf_ensemble.stack_method_ == ["predict_proba"] * 3

    def test_fit_raises_when_base_estimator_lacks_predict_proba(
        self, multiclass_wave_data
    ):
        X_train, _, y_train, _, features_group = multiclass_wave_data

        def extract_wave(wave, extract_indices=False):
            feature_indices = [group[wave] for group in features_group]
            if extract_indices:
                return None, None, feature_indices
            return None, None

        classifier = LongitudinalStackingClassifier(
            estimators=[
                ("alpha", NoPredictProbaClassifier()),
                ("beta", NoPredictProbaClassifier()),
                ("gamma", NoPredictProbaClassifier()),
            ],
            meta_learner=LogisticRegression(max_iter=200),
            extract_wave=extract_wave,
        )

        with pytest.raises(ValueError, match="must implement predict_proba"):
            classifier.fit(X_train, y_train)

    def test_missing_classes_in_stacking_cv_folds_still_return_global_probabilities(
        self,
    ):
        rng = np.random.RandomState(7)
        X = rng.normal(size=(21, 6))
        y = np.array([0] * 10 + [1] * 10 + [2])
        features_group = [[0, 1, 2], [3, 4, 5]]

        def extract_wave(wave, extract_indices=False):
            feature_indices = features_group[wave]
            if extract_indices:
                return None, None, feature_indices
            return None, None

        estimators = [
            (
                "wave_0",
                DecisionTreeClassifier(random_state=0).fit(X[:, features_group[0]], y),
            ),
            (
                "wave_1",
                DecisionTreeClassifier(random_state=1).fit(X[:, features_group[1]], y),
            ),
        ]

        classifier = LongitudinalStackingClassifier(
            estimators=estimators,
            meta_learner=LogisticRegression(max_iter=200),
            extract_wave=extract_wave,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The least populated class in y has only .*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Number of classes in training fold .*",
                category=RuntimeWarning,
            )
            classifier.fit(X, y)

        predictions = classifier.predict(X)
        probabilities = classifier.predict_proba(X)

        assert np.array_equal(classifier.classes_, np.array([0, 1, 2]))
        assert probabilities.shape == (len(X), len(classifier.classes_))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.all(np.isin(predictions, classifier.classes_))
        assert classifier.clf_ensemble.stack_method_ == ["predict_proba"] * len(
            estimators
        )
