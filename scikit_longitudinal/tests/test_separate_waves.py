import pytest
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.exceptions import NotFittedError

from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.data_preparation.separate_waves import SepWav


class TestSeparateWaves:
    @pytest.fixture
    def data(self):
        longitudinal_data = LongitudinalDataset("scikit_longitudinal/tests/dummy_data/dummy_data_3.csv")
        longitudinal_data.load_data_target_train_test_split(
            target_column="target_w2",
            remove_target_waves=True,
            target_wave_prefix="target_",
        )
        longitudinal_data.setup_features_group(input_data="elsa")

        return longitudinal_data

    @pytest.fixture
    def classifier(self):
        return RandomForestClassifier(max_depth=10, random_state=0)

    def test_validate_extract_wave_input(self, data, classifier):
        sepwav = SepWav(dataset=data, classifier=classifier)

        with pytest.raises(ValueError, match=r".*more than 0"):
            sepwav._extract_wave(-1)

    def test_validate_fit_input(self, data, classifier):
        with pytest.raises(ValueError, match=r".*classifier, dataset, and feature groups must not be None.*"):
            sepwav = SepWav(dataset=data, classifier=classifier)
            sepwav.classifier = None
            sepwav.fit(data.X_train, data.y_train)

    def test_fit_and_predict(self, data, classifier):
        sepwav = SepWav(dataset=data, classifier=classifier)

        sepwav.fit(data.X_train, data.y_train)

        y_pred = sepwav.predict(data.X_test)
        assert isinstance(y_pred, ndarray)
        assert all(y_pred == " B") or all(y_pred == " A")

        y_pred = sepwav.predict(data.X_test)
        assert isinstance(y_pred, ndarray)
        assert all(y_pred == " B") or all(y_pred == " A")

    def test_unfitted_predict(self, data, classifier):
        sepwav = SepWav(dataset=data, classifier=classifier)

        with pytest.raises(NotFittedError):
            sepwav.predict(data.X_test)

    def test_predict_with_wrong_X_type(self, data, classifier):
        sepwav = SepWav(dataset=data, classifier=classifier)

        sepwav.fit(data.X_train, data.y_train)

        with pytest.raises(ValueError):
            sepwav.predict("invalid_X")

    def test_fit_ensemble_strategy(self, data, classifier):
        sepwav = SepWav(dataset=data, classifier=classifier, ensemble_strategy="voting")
        sepwav.fit(data.X_train, data.y_train)
        assert isinstance(sepwav.clf_ensemble, VotingClassifier)

        sepwav = SepWav(dataset=data, classifier=classifier, ensemble_strategy="stacking")
        sepwav.fit(data.X_train, data.y_train)
        assert isinstance(sepwav.clf_ensemble, StackingClassifier)

        sepwav = SepWav(dataset=data, classifier=classifier, ensemble_strategy="invalid")
        with pytest.raises(ValueError, match=r".*Invalid ensemble strategy:*"):
            sepwav.fit(data.X_train, data.y_train)

    def test_parallelization(self, data, classifier):
        sepwav_parallel = SepWav(dataset=data, classifier=classifier, parallel=True)

        try:
            sepwav_parallel.fit(data.X_train, data.y_train)
        except Exception as e:
            pytest.fail(f"Parallel execution failed with error: {str(e)}")

    def test_predict_wave(self, data, classifier):
        sepwav = SepWav(dataset=data, classifier=classifier)

        sepwav.fit(data.X_train, data.y_train)

        for i in range(max(len(group) for group in data.feature_groups())):
            feature_indices = [group[i] for group in data.feature_groups() if i < len(group)]
            feature_indices.extend(data.non_longitudinal_features())

            X_test = data.X_test.iloc[:, feature_indices]
            y_pred = sepwav.predict_wave(i, X_test)
            assert isinstance(y_pred, ndarray)
            assert all(y_pred == " B") or all(y_pred == " A")

        with pytest.raises(IndexError):
            sepwav.predict_wave(-1, data.X_test)

        with pytest.raises(IndexError):
            sepwav.predict_wave(len(sepwav.classifiers), data.X_test)

        with pytest.raises(NotFittedError):
            sepwav.classifiers = None
            sepwav.predict_wave(3, data.X_test)

        with pytest.raises(NotFittedError):
            sepwav.classifiers = []
            sepwav.predict_wave(3, data.X_test)

    def test_invalid_wave_number_predict_wave(self, data, classifier):
        sepwav = SepWav(dataset=data, classifier=classifier)
        sepwav.fit(data.X_train, data.y_train)

        with pytest.raises(IndexError, match=r"Invalid wave number:.*"):
            sepwav.predict_wave(len(sepwav.classifiers), data.X_test)
