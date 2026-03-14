import numpy as np
import pytest
from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted

from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.estimators.ensemble.longitudinal_voting.custom_voting import (
    LongitudinalCustomVoting,
    TieBreaker,
)
from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import (
    LongitudinalEnsemblingStrategy,
    LongitudinalVotingClassifier,
)


class FixedPredictionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, predictions):
        self.predictions = tuple(predictions)

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        return self

    def predict(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X)
        return np.asarray(self.predictions[: len(X)])


class TestLongitudinalVotingClassifier:
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
    def multiclass_wave_data(self):
        X, y = make_classification(
            n_samples=180,
            n_features=6,
            n_informative=6,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            class_sep=1.5,
            random_state=7,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=7
        )
        features_group = [[0, 1, 2], [3, 4, 5]]
        return X_train, X_test, y_train, y_test, features_group

    def test_fit_no_estimators_raises_value_error(self, data):
        with pytest.raises(ValueError):
            LongitudinalVotingClassifier(estimators=[]).fit(data.X_train, data.y_train)

    def test_fit_invalid_voting_strategy_raises_value_error(
        self, data, valid_estimators
    ):
        with pytest.raises(ValueError):
            LongitudinalVotingClassifier(
                estimators=valid_estimators, voting="invalid"
            ).fit(data.X_train, data.y_train)

    def test_fit_with_valid_input_initializes_clf_ensemble(
        self, data, valid_estimators
    ):
        classifier = LongitudinalVotingClassifier(estimators=valid_estimators)
        classifier._fit(data.X_train, data.y_train)
        assert classifier.clf_ensemble is not None

    def test_predict_on_unfitted_model_raises_not_fitted_error(
        self, data, valid_estimators
    ):
        classifier = LongitudinalVotingClassifier(estimators=valid_estimators)
        with pytest.raises(NotFittedError):
            classifier._predict(data.X_test)

    def test_predict_proba_on_unfitted_model_raises_not_fitted_error(
        self, data, valid_estimators
    ):
        classifier = LongitudinalVotingClassifier(estimators=valid_estimators)
        with pytest.raises(NotFittedError):
            classifier._predict_proba(data.X_test)

    def test_predict_proba_on_fitted_model_returns_probabilities(
        self, data, valid_estimators
    ):
        classifier = LongitudinalVotingClassifier(estimators=valid_estimators)
        classifier._fit(data.X_train, data.y_train)
        probabilities = classifier._predict_proba(data.X_test)
        assert isinstance(probabilities, ndarray)

    def test_fit_majority_voting(self, data, valid_estimators):
        classifier = LongitudinalVotingClassifier(
            estimators=valid_estimators,
            voting=LongitudinalEnsemblingStrategy.MAJORITY_VOTING,
        )
        classifier._fit(data.X_train, data.y_train)
        assert classifier.clf_ensemble is not None

    def test_fit_decay_linear_voting(self, data, valid_estimators):
        classifier = LongitudinalVotingClassifier(
            estimators=valid_estimators,
            voting=LongitudinalEnsemblingStrategy.DECAY_LINEAR_VOTING,
        )
        classifier._fit(data.X_train, data.y_train)
        assert classifier.clf_ensemble is not None

    def test_fit_decay_exponential_voting(self, data, valid_estimators):
        classifier = LongitudinalVotingClassifier(
            estimators=valid_estimators,
            voting=LongitudinalEnsemblingStrategy.DECAY_EXPONENTIAL_VOTING,
        )
        classifier._fit(data.X_train, data.y_train)
        assert classifier.clf_ensemble is not None

    def test_fit_cv_based_voting(self, data, valid_estimators):
        classifier = LongitudinalVotingClassifier(
            estimators=valid_estimators,
            voting=LongitudinalEnsemblingStrategy.CV_BASED_VOTING,
        )
        classifier._fit(data.X_train, data.y_train)
        assert classifier.clf_ensemble is not None

    def test_calculate_linear_decay_weights(self):
        weights = LongitudinalVotingClassifier._calculate_linear_decay_weights(4)
        expected_weights = [1 / 10, 2 / 10, 3 / 10, 4 / 10]
        assert (
            weights == expected_weights
        ), "Linear decay weights calculation is incorrect"

    def test_calculate_exponential_decay_weights(self):
        weights = LongitudinalVotingClassifier._calculate_exponential_decay_weights(3)
        expected_weights = [
            np.exp(1) / sum(np.exp(j) for j in range(1, 4)),
            np.exp(2) / sum(np.exp(j) for j in range(1, 4)),
            np.exp(3) / sum(np.exp(j) for j in range(1, 4)),
        ]
        assert (
            weights == expected_weights
        ), "Exponential decay weights calculation is incorrect"

    def test_invalid_ensemble_strategy_raises_value_error(self, data, valid_estimators):
        with pytest.raises(ValueError):
            classifier = LongitudinalVotingClassifier(
                estimators=valid_estimators, voting="invalid_strategy"
            )
            classifier._fit(data.X_train, data.y_train)

    def test_multiclass_voting_supports_classes_predict_and_predict_proba(
        self, multiclass_wave_data
    ):
        X_train, X_test, y_train, _, features_group = multiclass_wave_data
        estimators = []
        for wave_index, estimator_name in enumerate(
            ["oldest-wave", "middle-wave", "latest-wave"]
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

        classifier = LongitudinalVotingClassifier(
            estimators=estimators,
            voting=LongitudinalEnsemblingStrategy.MAJORITY_VOTING,
            extract_wave=extract_wave,
        )
        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)

        assert np.array_equal(classifier.classes_, np.unique(y_train))
        assert probabilities.shape == (len(X_test), len(classifier.classes_))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.all(np.isin(predictions, classifier.classes_))

    def test_custom_voting_multiclass_tie_breaker_prefers_most_recent_vote(self):
        X = np.arange(3).reshape(-1, 1)
        y = np.array([0, 1, 2])
        estimators = [
            ("alpha", FixedPredictionClassifier([0, 0, 0]).fit(X, y)),
            ("beta", FixedPredictionClassifier([1, 1, 1]).fit(X, y)),
            ("gamma", FixedPredictionClassifier([2, 2, 2]).fit(X, y)),
        ]

        classifier = LongitudinalCustomVoting(
            estimators=estimators,
            tie_breaker=TieBreaker.LAST,
        )
        classifier.fit(X, y)

        predictions = classifier.predict(X)
        probabilities = classifier.predict_proba(X)

        assert np.array_equal(predictions, np.array([2, 2, 2]))
        assert np.allclose(probabilities, np.full((3, 3), 1.0 / 3.0))

    def test_custom_voting_multiclass_weighted_votes_return_labels_not_argmax_indices(
        self,
    ):
        X = np.arange(3).reshape(-1, 1)
        y = np.array([0, 1, 2])
        estimators = [
            ("first", FixedPredictionClassifier([0, 0, 0]).fit(X, y)),
            ("second", FixedPredictionClassifier([1, 1, 1]).fit(X, y)),
            ("third", FixedPredictionClassifier([2, 2, 2]).fit(X, y)),
        ]

        classifier = LongitudinalCustomVoting(
            estimators=estimators,
            weights=[0.1, 0.7, 0.2],
            tie_breaker=TieBreaker.FIRST,
        )
        classifier.fit(X, y)

        predictions = classifier.predict(X)
        probabilities = classifier.predict_proba(X)

        assert np.array_equal(classifier.classes_, np.array([0, 1, 2]))
        assert np.array_equal(predictions, np.array([1, 1, 1]))
        assert np.allclose(probabilities, np.tile(np.array([0.1, 0.7, 0.2]), (3, 1)))
