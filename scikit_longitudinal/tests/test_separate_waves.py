import pytest
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.data_preparation.separate_waves import SepWav
from scikit_longitudinal.estimators.ensemble.longitudinal_stacking.longitudinal_stacking import (
    LongitudinalStackingClassifier,
)
from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import (
    LongitudinalEnsemblingStrategy,
    LongitudinalVotingClassifier,
)


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
