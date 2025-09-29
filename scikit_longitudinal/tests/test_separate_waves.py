import pytest
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.data_preparation.separate_waves import SepWav
from scikit_longitudinal.estimators.ensemble.longitudinal_stacking.longitudinal_stacking import (
    LongitudinalStackingClassifier,
)
from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import (
    LongitudinalEnsemblingStrategy,
    LongitudinalVotingClassifier,
)
import numpy as np

class TestSeparateWaves:
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

        return longitudinal_data

    @pytest.fixture
    def classifier(self):
        return RandomForestClassifier(max_depth=10, random_state=0)

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
            feature_indices = [group[i] for group in data.feature_groups() if i < len(group)]
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
        with pytest.raises(ValueError, match=r"The classifier, dataset, and feature groups must not be None."):
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
        w = np.ones(len(data.y_train), dtype=float)
        sepwav = SepWav(
            estimator=classifier,
            features_group=data.feature_groups(),
            non_longitudinal_features=data.non_longitudinal_features(),
            feature_list_names=data.data.columns.tolist(),
            parallel=True,
        )
        with pytest.raises(ValueError, match=r"Sample weights are not supported in parallel mode"):
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
        assert all(getattr(estimator, "class_weight") == class_weight for _, estimator in sepwav.estimators)

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
        assert all(not hasattr(estimator, "class_weight") for _, estimator in sepwav.estimators)
