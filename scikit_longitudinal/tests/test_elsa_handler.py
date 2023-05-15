import os

import pytest

from scikit_longitudinal.data_preparation.elsa_handler import ElsaDataHandler


@pytest.fixture
def edh():
    return ElsaDataHandler("scikit_longitudinal/tests/dummy_data/dummy_elsa_data.csv", "core")


class TestELSAHandler:
    def test_get_unique_classes(self, edh):
        unique_classes = edh.get_unique_classes()
        assert sorted(unique_classes) == ["1", "2"]

    def test_create_datasets(self, edh):
        edh.create_datasets()

        assert "1" in edh.datasets
        assert "2" in edh.datasets

        class1_dataset = edh.datasets["1"]
        class2_dataset = edh.datasets["2"]

        assert [
            "feature1_w1",
            "feature1_w2",
            "feature2_w1",
            "feature2_w2",
            "class_1_w1",
            "class_1_w2",
        ] == class1_dataset.columns.to_list()
        assert [
            "feature1_w1",
            "feature1_w2",
            "feature2_w1",
            "feature2_w2",
            "class_2_w1",
            "class_2_w2",
        ] == class2_dataset.columns.to_list()

    def test_get_dataset(self, edh):
        edh.create_datasets()

        class1_dataset = edh.get_dataset("1")
        class2_dataset = edh.get_dataset("2")
        nonexistent_dataset = edh.get_dataset("3")

        assert class1_dataset is not None
        assert class2_dataset is not None
        assert nonexistent_dataset is None

    def test_wrong_elsa_type(self):
        with pytest.raises(ValueError):
            ElsaDataHandler("scikit_longitudinal/tests/dummy_data/dummy_elsa_data.csv", "wrong_type")

    def test_save_datasets(self, edh, tmp_path):
        edh.create_datasets()

        output_dir = tmp_path / "output"

        for file_format in ["csv", "arff"]:
            edh.save_datasets(dir_output=str(output_dir), file_format=file_format)
            class1_file = output_dir / f"1_dataset.{file_format}"
            class2_file = output_dir / f"2_dataset.{file_format}"

            assert class1_file.is_file()
            assert class2_file.is_file()
            os.remove(class1_file)
            os.remove(class2_file)

    def test_save_dataset_wrong_format(self, edh, tmp_path):
        edh.create_datasets()
        output_dir = tmp_path / "output"
        with pytest.raises(ValueError):
            edh.save_datasets(dir_output=str(output_dir), file_format="wrong_format")
