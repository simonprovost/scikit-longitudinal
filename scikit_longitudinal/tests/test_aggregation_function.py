import numpy as np
import pandas as pd
import pytest

from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.data_preparation.aggregation_function import AggFunc


class TestAggFunc:
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

    def test_init(self, data):
        with pytest.raises(ValueError, match=r"Invalid aggregation function:.*"):
            AggFunc(dataset=data, aggregation_func="invalid_func", parallel=False, num_cpus=1)

        data_copy = data
        data_copy._data.iloc[:, 0] = data_copy._data.iloc[:, 0].astype(str)
        with pytest.raises(ValueError, match=r"Inconsistent dtypes within feature group.*"):
            AggFunc(dataset=data_copy, aggregation_func="mean", parallel=False, num_cpus=1)

    def test_transform(self, data):
        agg_func = AggFunc(dataset=data, aggregation_func="mean", parallel=False, num_cpus=1)
        transformed_data = agg_func.transform()
        assert isinstance(transformed_data, LongitudinalDataset)

    def test_transform_parallel(self, data):
        agg_func_parallel = AggFunc(dataset=data, aggregation_func="mean", parallel=True, num_cpus=2)
        transformed_data_parallel = agg_func_parallel.transform()
        assert isinstance(transformed_data_parallel, LongitudinalDataset)

    def test_same_output_parallel_and_without(self, data):
        agg_func = AggFunc(dataset=data, aggregation_func="mean", parallel=False, num_cpus=1)
        transformed_data = agg_func.transform()

        longitudinal_data = LongitudinalDataset("scikit_longitudinal/tests/dummy_data/dummy_data_3.csv")
        longitudinal_data.load_data_target_train_test_split(
            target_column="target_w2",
            remove_target_waves=True,
            target_wave_prefix="target_",
        )
        longitudinal_data.setup_features_group(input_data="elsa")

        agg_func_parallel = AggFunc(dataset=longitudinal_data, aggregation_func="mean", parallel=True, num_cpus=2)
        transformed_data_parallel = agg_func_parallel.transform()

        assert transformed_data.data.equals(transformed_data_parallel.data)

    def test_transform_mean(self, data):
        agg_func = AggFunc(dataset=data, aggregation_func="mean", parallel=False, num_cpus=1)
        transformed_data = agg_func.transform()
        assert isinstance(transformed_data, LongitudinalDataset)

    def test_transform_median(self, data):
        agg_func = AggFunc(dataset=data, aggregation_func="median", parallel=False, num_cpus=1)
        transformed_data = agg_func.transform()
        assert isinstance(transformed_data, LongitudinalDataset)

    def test_transform_categorical_feature(self, data):
        tmp = pd.DataFrame(
            {
                "age_w1": np.repeat([20, 30], [19, 21]),
                "income_w1": np.repeat([50000, 60000], [19, 21]),
                "income_w2": np.repeat([52000, 62000], [19, 21]),
                "income_w3": np.repeat([54000, 64000], [19, 21]),
                "age_w2": np.repeat([21, 31], [19, 21]),
                "age_w3": np.repeat([22, 32], [19, 21]),
                "target_w1": ["A"] * 40,
                "target_w2": ["B"] * 40,
                "gender": np.repeat([1, 0], [19, 21]),
            }
        )

        data_copy = tmp.copy()

        data_copy["age_w1"] = data_copy["age_w1"].astype(str)
        data_copy["age_w2"] = data_copy["age_w2"].astype(str)
        data_copy["age_w3"] = data_copy["age_w3"].astype(str)
        data_copy["age_w1"].replace({"20": "twenty", "30": "thirty"}, inplace=True)
        data_copy["age_w2"].replace({"21": "twentyone", "31": "thirtyone"}, inplace=True)
        data_copy["age_w3"].replace({"22": "twentytwo", "32": "thirtytwo"}, inplace=True)
        data.set_data(data_copy)

        with pytest.warns(
            UserWarning,
            match=r"Aggregation function is .* but feature group .* is categorical. Using " r"mode instead.",
        ):
            agg_func = AggFunc(dataset=data, aggregation_func="mean", parallel=False, num_cpus=1)
            transformed_data = agg_func.transform()

        expected_result = data_copy[["age_w1", "age_w2", "age_w3"]].mode(axis=1)[0]
        transformed_result = transformed_data.data["agg_mode_age"]
        assert transformed_result.equals(expected_result), f"Expected {expected_result} but got {transformed_result}"

    def test_init_with_custom_aggregation_func(self, data):
        def custom_agg_func(x):
            return x.sum()

        agg_func = AggFunc(dataset=data, aggregation_func=custom_agg_func, parallel=False, num_cpus=1)
        assert agg_func.aggregation_func == custom_agg_func
        assert agg_func.agg_func == custom_agg_func

    def test_init_with_invalid_aggregation_func(self, data):
        with pytest.raises(ValueError, match=r"aggregation_func must be either a string.*or a function."):
            AggFunc(dataset=data, aggregation_func=42, parallel=False, num_cpus=1)
