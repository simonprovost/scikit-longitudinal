import pandas as pd
import pytest

from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.data_preparation.aggregation_function import AggrFunc


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
            AggrFunc(
                aggregation_func="invalid_func",
                features_group=data.feature_groups(),
                non_longitudinal_features=data.non_longitudinal_features(),
                feature_list_names=data.data.columns.tolist(),
            )

    def test_transform(self, data):
        agg_func = AggrFunc(
            aggregation_func="mean",
            features_group=data.feature_groups(),
            non_longitudinal_features=data.non_longitudinal_features(),
            feature_list_names=data.data.columns.tolist(),
        )
        agg_func.prepare_data(data.data.to_numpy())
        transformed_dataset, feature_groups, non_longitudinal_features, feature_names = agg_func._transform()

        assert isinstance(transformed_dataset, pd.DataFrame)
        assert feature_groups is None
        assert non_longitudinal_features is None
        assert isinstance(feature_names, list)
        assert feature_names == [" gender", "agg_mean_age", "agg_mean_income"]

    def test_transform_parallel(self, data):
        agg_func_parallel = AggrFunc(
            aggregation_func="mean",
            features_group=data.feature_groups(),
            non_longitudinal_features=data.non_longitudinal_features(),
            feature_list_names=data.data.columns.tolist(),
            parallel=True,
            num_cpus=2,
        )
        agg_func_parallel.prepare_data(data.data.to_numpy())
        (
            transformed_dataset_parallel,
            feature_groups_parallel,
            non_longitudinal_features_parallel,
            feature_names_parallel,
        ) = agg_func_parallel._transform()

        assert isinstance(transformed_dataset_parallel, pd.DataFrame)
        assert feature_groups_parallel is None
        assert non_longitudinal_features_parallel is None
        assert isinstance(feature_names_parallel, list)
        assert feature_names_parallel == [" gender", "agg_mean_age", "agg_mean_income"]

    def test_same_output_parallel_and_without(self, data):
        agg_func = AggrFunc(
            aggregation_func="mean",
            features_group=data.feature_groups(),
            non_longitudinal_features=data.non_longitudinal_features(),
            feature_list_names=data.data.columns.tolist(),
        )
        agg_func.prepare_data(data.data.to_numpy())
        transformed_dataset, _, _, _ = agg_func._transform()

        agg_func_parallel = AggrFunc(
            aggregation_func="mean",
            features_group=data.feature_groups(),
            non_longitudinal_features=data.non_longitudinal_features(),
            feature_list_names=data.data.columns.tolist(),
            parallel=True,
            num_cpus=2,
        )
        agg_func_parallel.prepare_data(data.data.to_numpy())
        transformed_dataset_parallel, _, _, _ = agg_func_parallel._transform()

        assert transformed_dataset.equals(transformed_dataset_parallel)

    def test_init_with_custom_aggregation_func(self, data):
        def custom_agg_func(x):
            return x.sum()

        agg_func = AggrFunc(
            aggregation_func=custom_agg_func,
            features_group=data.feature_groups(),
            non_longitudinal_features=data.non_longitudinal_features(),
            feature_list_names=data.data.columns.tolist(),
        )
        assert agg_func.aggregation_func == custom_agg_func
        assert agg_func.agg_func == custom_agg_func

    def test_init_with_invalid_aggregation_func(self, data):
        with pytest.raises(ValueError, match=r"aggregation_func must be either a string.*or a function."):
            AggrFunc(
                aggregation_func=42,
                features_group=data.feature_groups(),
                non_longitudinal_features=data.non_longitudinal_features(),
                feature_list_names=data.data.columns.tolist(),
            )
