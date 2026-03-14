import importlib.util

import numpy as np
import pytest
from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.data_preparation.separate_waves import SepWav
from scikit_longitudinal.estimators.ensemble.longitudinal_stacking.longitudinal_stacking import (
    LongitudinalStackingClassifier,
)
from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import (
    LongitudinalEnsemblingStrategy,
    LongitudinalVotingClassifier,
)

RAY_AVAILABLE = importlib.util.find_spec("ray") is not None


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


class TestSeparateWaves:
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

        return longitudinal_data

    @pytest.fixture
    def classifier(self):
        return RandomForestClassifier(max_depth=10, random_state=0)

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
            random_state=23,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=23
        )
        features_group = [[0, 1, 2], [3, 4, 5]]
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        return X_train, X_test, y_train, y_test, features_group, feature_names

    @pytest.fixture
    def sepwav(self, data, classifier):
        return SepWav(
            estimator=classifier,
            features_group=data.feature_groups(),
            non_longitudinal_features=data.non_longitudinal_features(),
            feature_list_names=data.data.columns.tolist(),
        )

    def test_fit_and_predict(self, data, sepwav):
        sepwav.fit(data.X_train, data.y_train)

        y_pred = sepwav.predict(data.X_test)
        assert isinstance(y_pred, ndarray)
        assert all(y_pred == 0) or all(y_pred == 1)

    def test_unfitted_predict(self, data, sepwav):
        with pytest.raises(NotFittedError):
            sepwav.predict(data.X_test)

    def test_predict_with_wrong_X_type(self, data, sepwav):
        sepwav.fit(data.X_train, data.y_train)

        with pytest.raises(ValueError):
            sepwav.predict("invalid_X")

    def test_fit_ensemble_strategy(self, data, classifier):
        y_train = data.y_train.copy()
        for i in range(len(y_train) // 2):
            y_train[i] += 1

        x_test = data.X_test.copy()
        for i in range(len(x_test) // 2):
            x_test.iloc[i, 0] += 1

        sepwav_voting = SepWav(
            estimator=classifier,
            features_group=data.feature_groups(),
            non_longitudinal_features=data.non_longitudinal_features(),
            feature_list_names=data.data.columns.tolist(),
            voting=LongitudinalEnsemblingStrategy.MAJORITY_VOTING,
        )
        sepwav_voting.fit(data.X_train, y_train)
        assert isinstance(sepwav_voting.clf_ensemble, LongitudinalVotingClassifier)

        sepwav_stacking = SepWav(
            estimator=classifier,
            features_group=data.feature_groups(),
            non_longitudinal_features=data.non_longitudinal_features(),
            feature_list_names=data.data.columns.tolist(),
            voting=LongitudinalEnsemblingStrategy.STACKING,
        )
        sepwav_stacking.fit(data.X_train, y_train)
        assert isinstance(sepwav_stacking.clf_ensemble, LongitudinalStackingClassifier)

        sepwav_invalid = SepWav(
            estimator=classifier,
            features_group=data.feature_groups(),
            non_longitudinal_features=data.non_longitudinal_features(),
            feature_list_names=data.data.columns.tolist(),
            voting="invalid",
        )
        with pytest.raises(ValueError, match=r".*Invalid ensemble strategy:*"):
            sepwav_invalid.fit(data.X_train, y_train)

    def test_parallelization(self, data, classifier):
        if not RAY_AVAILABLE:
            pytest.skip(
                "Ray not installed; install Scikit-longitudinal[parallelisation] to run parallel tests."
            )

        sepwav_parallel = SepWav(
            estimator=classifier,
            features_group=data.feature_groups(),
            non_longitudinal_features=data.non_longitudinal_features(),
            feature_list_names=data.data.columns.tolist(),
            parallel=True,
        )
        try:
            sepwav_parallel.fit(data.X_train, data.y_train)
        except Exception as e:
            pytest.fail(f"Parallel execution failed with error: {str(e)}")

    def test_predict_wave(self, data, sepwav):
        sepwav.fit(data.X_train, data.y_train)

        for i in range(len(sepwav.features_group)):
            feature_indices = [
                group[i] for group in data.feature_groups() if i < len(group)
            ]
            feature_indices.extend(data.non_longitudinal_features())
            X_test = data.X_test.iloc[:, feature_indices]
            y_pred = sepwav.predict_wave(i, X_test)
            assert isinstance(y_pred, ndarray)
            assert all(y_pred == 0) or all(y_pred == 1)

        with pytest.raises(IndexError):
            sepwav.predict_wave(-1, data.X_test)

        with pytest.raises(IndexError):
            sepwav.predict_wave(len(sepwav.estimators), data.X_test)

        with pytest.raises(NotFittedError):
            sepwav.estimators = None
            sepwav.predict_wave(3, data.X_test)

        with pytest.raises(NotFittedError):
            sepwav.estimators = []
            sepwav.predict_wave(3, data.X_test)

    def test_invalid_wave_number_predict_wave(self, data, sepwav):
        sepwav.fit(data.X_train, data.y_train)
        with pytest.raises(IndexError, match=r"Invalid wave number:.*"):
            sepwav.predict_wave(len(sepwav.estimators), data.X_test)

    def test_validate_extract_wave_input(self, sepwav):
        with pytest.raises(ValueError, match=r".*more than 0"):
            sepwav._extract_wave(wave=-1)

    def test_validate_fit_input(self, data, sepwav):
        with pytest.raises(
            ValueError,
            match=r"The classifier, dataset, and feature groups must not be None.",
        ):
            sepwav.estimator = None
            sepwav.fit(data.X_train, data.y_train)

    def test_fit_with_sample_weight_voting(self, data, sepwav):
        w = np.ones(len(data.y_train), dtype=float)
        w[: len(w) // 2] = 3.0  # non-uniform weights
        sepwav.fit(data.X_train, data.y_train, sample_weight=w)
        assert isinstance(sepwav.clf_ensemble, LongitudinalVotingClassifier)
        y_pred = sepwav.predict(data.X_test)
        assert isinstance(y_pred, ndarray)

    def test_fit_with_sample_weight_stacking(self, data, classifier):
        w = np.linspace(0.5, 2.0, num=len(data.y_train))
        y_train = np.asarray(data.y_train).copy()
        if len(np.unique(y_train)) < 2:
            half = len(y_train) // 2
            y_train[:half] = 1 - y_train[:half]

        sepwav = SepWav(
            estimator=classifier,
            features_group=data.feature_groups(),
            non_longitudinal_features=data.non_longitudinal_features(),
            feature_list_names=data.data.columns.tolist(),
            voting=LongitudinalEnsemblingStrategy.STACKING,
        )
        sepwav.fit(data.X_train, y_train, sample_weight=w)
        assert isinstance(sepwav.clf_ensemble, LongitudinalStackingClassifier)
        y_pred = sepwav.predict(data.X_test)
        assert isinstance(y_pred, ndarray)

    def test_fit_with_sample_weight_parallel_error(self, data, classifier):
        if not RAY_AVAILABLE:
            pytest.skip(
                "Ray not installed; install Scikit-longitudinal[parallelisation] to run parallel tests."
            )

        w = np.ones(len(data.y_train), dtype=float)
        sepwav = SepWav(
            estimator=classifier,
            features_group=data.feature_groups(),
            non_longitudinal_features=data.non_longitudinal_features(),
            feature_list_names=data.data.columns.tolist(),
            parallel=True,
        )
        with pytest.raises(
            ValueError, match=r"Sample weights are not supported in parallel mode"
        ):
            sepwav.fit(data.X_train, data.y_train, sample_weight=w)

    def test_fit_with_mismatched_sample_weight_raises(self, data, sepwav):
        # Wrong length should bubble up from the underlying estimator
        bad_w = np.ones(len(data.y_train) - 1, dtype=float)
        with pytest.raises(ValueError):
            sepwav.fit(data.X_train, data.y_train, sample_weight=bad_w)

    def test_predict_proba_after_weighted_fit(self, data, sepwav):
        w = np.ones(len(data.y_train), dtype=float)
        w[len(w) // 3 : 2 * len(w) // 3] = 2.0
        sepwav.fit(data.X_train, data.y_train, sample_weight=w)

        proba = sepwav.predict_proba(data.X_test)
        assert isinstance(proba, ndarray)
        assert proba.ndim == 2
        assert proba.shape[0] == len(data.X_test)
        assert proba.shape[1] >= 1

    def test_class_weight_propagated_to_supported_estimators(self, data):
        class_weight = {0: 1.0, 1: 3.5}
        sepwav = SepWav(
            estimator=DecisionTreeClassifier(random_state=0),
            features_group=data.feature_groups(),
            non_longitudinal_features=data.non_longitudinal_features(),
            feature_list_names=data.data.columns.tolist(),
            class_weight=class_weight,
        )

        sepwav.fit(data.X_train, data.y_train)

        assert sepwav.class_weight == class_weight
        assert all(
            getattr(estimator, "class_weight") == class_weight
            for _, estimator in sepwav.estimators
        )

    def test_class_weight_skipped_for_unsupported_estimators(self, data):
        class_weight = {0: 2.0, 1: 1.0}
        sepwav = SepWav(
            estimator=KNeighborsClassifier(),
            features_group=data.feature_groups(),
            non_longitudinal_features=data.non_longitudinal_features(),
            feature_list_names=data.data.columns.tolist(),
            class_weight=class_weight,
        )

        sepwav.fit(data.X_train, data.y_train)

        assert sepwav.class_weight == class_weight
        assert all(
            not hasattr(estimator, "class_weight") for _, estimator in sepwav.estimators
        )

    def test_class_weight_propagated_to_stacking_meta_learner(self, data):
        class_weight = "balanced"
        meta_learner = LogisticRegression(max_iter=50)
        sepwav = SepWav(
            estimator=DecisionTreeClassifier(random_state=0),
            features_group=data.feature_groups(),
            non_longitudinal_features=data.non_longitudinal_features(),
            feature_list_names=data.data.columns.tolist(),
            voting=LongitudinalEnsemblingStrategy.STACKING,
            stacking_meta_learner=meta_learner,
            class_weight=class_weight,
        )
        y_train = data.y_train.copy()
        for i in range(len(y_train) // 2):
            y_train[i] += 1

        sepwav.fit(data.X_train, y_train)

        assert isinstance(sepwav.clf_ensemble, LongitudinalStackingClassifier)
        assert sepwav.class_weight == class_weight
        assert all(
            getattr(estimator, "class_weight") == class_weight
            for _, estimator in sepwav.estimators
        )
        assert (
            getattr(sepwav.clf_ensemble.meta_learner, "class_weight", None)
            == class_weight
        )

    def test_multiclass_majority_voting_end_to_end(self, multiclass_wave_data):
        X_train, X_test, y_train, _, features_group, feature_names = (
            multiclass_wave_data
        )
        sepwav = SepWav(
            estimator=RandomForestClassifier(max_depth=5, random_state=0),
            features_group=features_group,
            non_longitudinal_features=[],
            feature_list_names=feature_names,
            voting=LongitudinalEnsemblingStrategy.MAJORITY_VOTING,
        )

        sepwav.fit(X_train, y_train)
        predictions = sepwav.predict(X_test)
        probabilities = sepwav.predict_proba(X_test)

        assert np.array_equal(sepwav.classes_, np.unique(y_train))
        assert probabilities.shape == (len(X_test), len(sepwav.classes_))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.all(np.isin(predictions, sepwav.classes_))

    def test_multiclass_stacking_end_to_end(self, multiclass_wave_data):
        X_train, X_test, y_train, _, features_group, feature_names = (
            multiclass_wave_data
        )
        sepwav = SepWav(
            estimator=RandomForestClassifier(max_depth=5, random_state=0),
            features_group=features_group,
            non_longitudinal_features=[],
            feature_list_names=feature_names,
            voting=LongitudinalEnsemblingStrategy.STACKING,
            stacking_meta_learner=LogisticRegression(max_iter=200),
        )

        sepwav.fit(X_train, y_train)
        predictions = sepwav.predict(X_test)
        probabilities = sepwav.predict_proba(X_test)

        assert np.array_equal(sepwav.classes_, np.unique(y_train))
        assert probabilities.shape == (len(X_test), len(sepwav.classes_))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.all(np.isin(predictions, sepwav.classes_))
        assert sepwav.clf_ensemble.clf_ensemble.stack_method_ == [
            "predict_proba"
        ] * len(sepwav.estimators)

    def test_stacking_uses_wave_specific_features_inside_separate_waves(
        self, multiclass_wave_data
    ):
        X_train, X_test, y_train, _, features_group, feature_names = (
            multiclass_wave_data
        )
        sepwav = SepWav(
            estimator=FeatureCountClassifier(expected_n_features=2),
            features_group=features_group,
            non_longitudinal_features=[],
            feature_list_names=feature_names,
            voting=LongitudinalEnsemblingStrategy.STACKING,
            stacking_meta_learner=LogisticRegression(max_iter=200),
        )

        sepwav.fit(X_train, y_train)
        predictions = sepwav.predict(X_test)
        probabilities = sepwav.predict_proba(X_test)

        assert predictions.shape == (len(X_test),)
        assert probabilities.shape == (len(X_test), len(sepwav.classes_))
        assert sepwav.clf_ensemble.clf_ensemble.stack_method_ == [
            "predict_proba"
        ] * len(sepwav.estimators)

    def test_stacking_requires_probability_capable_base_estimators(
        self, multiclass_wave_data
    ):
        X_train, _, y_train, _, features_group, feature_names = multiclass_wave_data
        sepwav = SepWav(
            estimator=NoPredictProbaClassifier(),
            features_group=features_group,
            non_longitudinal_features=[],
            feature_list_names=feature_names,
            voting=LongitudinalEnsemblingStrategy.STACKING,
            stacking_meta_learner=LogisticRegression(max_iter=200),
        )

        with pytest.raises(ValueError, match="must implement predict_proba"):
            sepwav.fit(X_train, y_train)
