import numpy as np
import pytest
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import (
    LongitudinalEnsemblingStrategy,
    LongitudinalVotingClassifier,
)


class TestLongitudinalVotingClassifier:
    @pytest.fixture
    def data(self):
        longitudinal_data = LongitudinalDataset("scikit_longitudinal/tests/dummy_data/dummy_data_3.csv")
        longitudinal_data.load_data_target_train_test_split(
            target_column="target_w2",
            remove_target_waves=True,
            target_wave_prefix="target_",
        )
        longitudinal_data.sety_train([0 if y == " A" else 1 for y in longitudinal_data.y_train])
        longitudinal_data.sety_test([0 if y == " A" else 1 for y in longitudinal_data.y_test])
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
        return [(f"estimator_{i}", estimator.fit(data.X_train, data.y_train)) for i in range(3)]

    def test_fit_no_estimators_raises_value_error(self, data):
        with pytest.raises(ValueError):
            LongitudinalVotingClassifier(estimators=[]).fit(data.X_train, data.y_train)

    def test_fit_invalid_voting_strategy_raises_value_error(self, data, valid_estimators):
        with pytest.raises(ValueError):
            LongitudinalVotingClassifier(estimators=valid_estimators, voting="invalid").fit(data.X_train, data.y_train)

    def test_fit_with_valid_input_initializes_clf_ensemble(self, data, valid_estimators):
        classifier = LongitudinalVotingClassifier(estimators=valid_estimators)
        classifier._fit(data.X_train, data.y_train)
        assert classifier.clf_ensemble is not None

    def test_predict_on_unfitted_model_raises_not_fitted_error(self, data, valid_estimators):
        classifier = LongitudinalVotingClassifier(estimators=valid_estimators)
        with pytest.raises(NotFittedError):
            classifier._predict(data.X_test)

    def test_predict_proba_on_unfitted_model_raises_not_fitted_error(self, data, valid_estimators):
        classifier = LongitudinalVotingClassifier(estimators=valid_estimators)
        with pytest.raises(NotFittedError):
            classifier._predict_proba(data.X_test)

    def test_predict_proba_on_fitted_model_returns_probabilities(self, data, valid_estimators):
        classifier = LongitudinalVotingClassifier(estimators=valid_estimators)
        classifier._fit(data.X_train, data.y_train)
        probabilities = classifier._predict_proba(data.X_test)
        assert isinstance(probabilities, ndarray)

    def test_fit_majority_voting(self, data, valid_estimators):
        classifier = LongitudinalVotingClassifier(
            estimators=valid_estimators, voting=LongitudinalEnsemblingStrategy.MAJORITY_VOTING
        )
        classifier._fit(data.X_train, data.y_train)
        assert classifier.clf_ensemble is not None

    def test_fit_decay_linear_voting(self, data, valid_estimators):
        classifier = LongitudinalVotingClassifier(
            estimators=valid_estimators, voting=LongitudinalEnsemblingStrategy.DECAY_LINEAR_VOTING
        )
        classifier._fit(data.X_train, data.y_train)
        assert classifier.clf_ensemble is not None

    def test_fit_decay_exponential_voting(self, data, valid_estimators):
        classifier = LongitudinalVotingClassifier(
            estimators=valid_estimators, voting=LongitudinalEnsemblingStrategy.DECAY_EXPONENTIAL_VOTING
        )
        classifier._fit(data.X_train, data.y_train)
        assert classifier.clf_ensemble is not None

    def test_fit_cv_based_voting(self, data, valid_estimators):
        classifier = LongitudinalVotingClassifier(
            estimators=valid_estimators, voting=LongitudinalEnsemblingStrategy.CV_BASED_VOTING
        )
        classifier._fit(data.X_train, data.y_train)
        assert classifier.clf_ensemble is not None

    def test_calculate_linear_decay_weights(self):
        weights = LongitudinalVotingClassifier._calculate_linear_decay_weights(4)
        expected_weights = [1 / 10, 2 / 10, 3 / 10, 4 / 10]
        assert weights == expected_weights, "Linear decay weights calculation is incorrect"

    def test_calculate_exponential_decay_weights(self):
        weights = LongitudinalVotingClassifier._calculate_exponential_decay_weights(3)
        expected_weights = [
            np.exp(1) / sum(np.exp(j) for j in range(1, 4)),
            np.exp(2) / sum(np.exp(j) for j in range(1, 4)),
            np.exp(3) / sum(np.exp(j) for j in range(1, 4)),
        ]
        assert weights == expected_weights, "Exponential decay weights calculation is incorrect"

    def test_invalid_ensemble_strategy_raises_value_error(self, data, valid_estimators):
        with pytest.raises(ValueError):
            classifier = LongitudinalVotingClassifier(estimators=valid_estimators, voting="invalid_strategy")
            classifier._fit(data.X_train, data.y_train)
