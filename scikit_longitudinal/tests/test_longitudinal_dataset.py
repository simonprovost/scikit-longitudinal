from pathlib import Path

import pytest

from scikit_longitudinal.data_preparation import LongitudinalDataset

CSV_DATA = Path("scikit_longitudinal/tests/dummy_data/dummy_data.csv")
CSV_DATA_1 = Path("scikit_longitudinal/tests/dummy_data/dummy_data_1.csv")
CSV_DATA_2 = Path("scikit_longitudinal/tests/dummy_data/dummy_data_2.csv")

ARFF_DATA = Path("scikit_longitudinal/tests/dummy_data/dummy_data.arff")
ARFF_DATA_1 = Path("scikit_longitudinal/tests/dummy_data/dummy_data_1.arff")
ARFF_DATA_2 = Path("scikit_longitudinal/tests/dummy_data/dummy_data_2.arff")


@pytest.fixture(params=[CSV_DATA, ARFF_DATA])
def dataset(request):
    ds = LongitudinalDataset(request.param)
    ds.load_data()
    return ds


@pytest.fixture(params=[CSV_DATA_1, ARFF_DATA_1])
def dataset_1(request):
    ds = LongitudinalDataset(request.param)
    ds.load_data()
    return ds


@pytest.fixture(params=[CSV_DATA_2, ARFF_DATA_2])
def dataset_2(request):
    ds = LongitudinalDataset(request.param)
    ds.load_data()
    return ds


def _check_list_type(arg0, arg1):
    assert arg0 is not None
    assert isinstance(arg0, list)
    assert all(isinstance(name, arg1) for name in arg0)


class TestLongitudinalDataset:
    def test_setup_elsa_feature_groups(self, dataset):
        dataset.setup_features_group("Elsa")
        assert dataset._feature_groups is not None
        assert len(dataset._feature_groups) > 0

    def test_fail_setup_elsa_feature_group_with_one_wave(self, dataset):
        dataset._data.drop(["feature1_w2"], inplace=True, axis=1)
        with pytest.raises(ValueError):
            dataset.setup_features_group("Elsa")

    def test_fail_setup_invalid_input_data(self, dataset):
        with pytest.raises(ValueError):
            dataset.setup_features_group("Invalid")

    def test_setup_features_group_with_indices(self, dataset):
        feature_groups = [[0, 1], [2, 3]]
        dataset.setup_features_group(feature_groups)
        assert dataset._feature_groups == feature_groups

    def test_setup_features_group_with_names(self, dataset):
        feature_groups = [["feature1_w1", "feature1_w2"], ["feature2_w1", "feature2_w2"]]
        dataset.setup_features_group(feature_groups)
        assert dataset._feature_groups == [[0, 1], [2, 3]]

    def test_fail_setup_features_group_with_mixed_types(self, dataset):
        with pytest.raises(ValueError):
            feature_groups = [["feature1_w1", 1], ["feature2_w1", "feature2_w2"]]
            dataset.setup_features_group(feature_groups)

    def test_convert_feature_names_to_indices(self, dataset):
        feature_groups = [["feature1_w1", "feature1_w2"], ["feature2_w1", "feature2_w2"]]
        converted_feature_groups = dataset._convert_feature_names_to_indices(feature_groups)
        assert converted_feature_groups == [[0, 1], [2, 3]]

    def test_fail_convert_feature_names_to_indices_with_invalid_names(self, dataset):
        with pytest.raises(ValueError):
            feature_groups = [["nonexistent_feature", "feature1_w2"], ["feature2_w1", "feature2_w2"]]
            dataset._convert_feature_names_to_indices(feature_groups)

    def test_elsa_feature_groups_dummy_data_1(self, dataset_1):
        dataset_1.load_data_target_train_test_split(
            target_column="target_w2",
            remove_target_waves=True,
            target_wave_prefix="target_",
            test_size=0.2,
            random_state=42,
        )
        dataset_1.setup_features_group("Elsa")
        assert dataset_1._feature_groups is not None
        assert len(dataset_1._feature_groups) > 0
        assert dataset_1._feature_groups == [[0, 4, 5], [1, 2, 3]]

    def test_elsa_feature_groups_dummy_data_2(self, dataset_2):
        dataset_2.setup_features_group("Elsa")
        assert dataset_2._feature_groups is not None
        assert len(dataset_2._feature_groups) > 0
        assert dataset_2._feature_groups == [[0, 4, 5], [1, 2, 3]]

    def test_fail_with_fake_file(self):
        with pytest.raises(FileNotFoundError):
            LongitudinalDataset("fake_file.csv")

    def test_load_target_valid_column(self, dataset):
        dataset.load_target(target_column="target_w2")
        assert dataset._target is not None
        assert "target_w2" not in dataset._data.columns
        assert "target_w1" in dataset._data.columns

    def test_fail_load_target_invalid_column(self, dataset):
        with pytest.raises(ValueError):
            dataset.load_target(target_column="nonexistent_column")

    def test_fail_load_target_no_data_loaded(self):
        empty_dataset = LongitudinalDataset(CSV_DATA)
        with pytest.raises(ValueError):
            empty_dataset.load_target(target_column="target_w2")

    def test_load_target_remove_target_waves(self, dataset):
        dataset.load_target(target_column="target_w2", target_wave_prefix="target_", remove_target_waves=True)
        assert dataset._target is not None
        assert "target_w1" not in dataset._data.columns

    def test_load_train_test_split(self, dataset):
        dataset.load_target(target_column="target_w2")
        dataset.load_train_test_split(test_size=0.2, random_state=42)
        assert dataset._X_train is not None
        assert dataset._X_test is not None
        assert dataset._y_train is not None
        assert dataset._y_test is not None

    def test_property_methods(self, dataset):
        dataset.load_target(target_column="target_w2")
        dataset.load_train_test_split(test_size=0.2, random_state=42)
        assert dataset.target.equals(dataset._target)
        assert dataset.X_train.equals(dataset._X_train)
        assert dataset.X_test.equals(dataset._X_test)
        assert dataset.y_train.equals(dataset._y_train)
        assert dataset.y_test.equals(dataset._y_test)

    def test_setter_methods(self, dataset):
        dataset.load_target(target_column="target_w2")
        dataset.load_train_test_split(test_size=0.2, random_state=42)

        new_data = dataset._data.copy()
        new_target = dataset._target.copy()
        new_X_train = dataset._X_train.copy()
        new_X_test = dataset._X_test.copy()
        new_y_train = dataset._y_train.copy()
        new_y_test = dataset._y_test.copy()

        dataset.set_data(new_data)
        dataset.set_target(new_target)
        dataset.setX_train(new_X_train)
        dataset.setX_test(new_X_test)
        dataset.sety_train(new_y_train)
        dataset.sety_test(new_y_test)

        assert dataset._data.equals(new_data)
        assert dataset._target.equals(new_target)
        assert dataset._X_train.equals(new_X_train)
        assert dataset._X_test.equals(new_X_test)
        assert dataset._y_train.equals(new_y_train)
        assert dataset._y_test.equals(new_y_test)

    def test_load_data_target_train_test_split(self, dataset):
        dataset.load_data_target_train_test_split(target_column="target_w2", test_size=0.2, random_state=42)
        assert dataset._data is not None
        assert dataset._target is not None
        assert dataset._X_train is not None
        assert dataset._X_test is not None
        assert dataset._y_train is not None
        assert dataset._y_test is not None

    def test_non_longitudinal_features_names(self, dataset_1):
        dataset_1.load_data_target_train_test_split(
            target_column="target_w2",
            remove_target_waves=True,
            target_wave_prefix="target_",
            test_size=0.2,
            random_state=42,
        )
        dataset_1.setup_features_group("Elsa")
        non_longitudinal_feature_names = dataset_1.non_longitudinal_features(names=True)
        non_longitudinal_feature_indices = dataset_1.non_longitudinal_features(names=False)
        _check_list_type(non_longitudinal_feature_names, str)
        _check_list_type(non_longitudinal_feature_indices, int)

    def test_fail_load_train_test_split_no_data_or_target(self, dataset):
        dataset.load_data()
        dataset._data = None
        with pytest.raises(ValueError, match="No data or target is loaded. Load them first."):
            dataset.load_train_test_split(test_size=0.2, random_state=42)

    def test_feature_groups_indices(self, dataset):
        dataset.load_data()
        dataset.setup_features_group([["feature1_w1", "feature1_w2"], ["feature2_w1", "feature2_w2"]])
        feature_groups = dataset.feature_groups()
        assert feature_groups == [[0, 1], [2, 3]]

    def test_feature_groups_names(self, dataset):
        dataset.load_data()
        dataset.setup_features_group([["feature1_w1", "feature1_w2"], ["feature2_w1", "feature2_w2"]])
        feature_groups = dataset.feature_groups(names=True)
        assert feature_groups == [["feature1_w1", "feature1_w2"], ["feature2_w1", "feature2_w2"]]
