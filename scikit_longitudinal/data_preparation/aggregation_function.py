import re
import warnings
from typing import Any, Callable, List, Union

import numpy as np
import pandas as pd
import ray
from scipy import stats

from scikit_longitudinal.data_preparation import LongitudinalDataset

AGG_FUNCS = {  # pragma: no cover
    "mean": np.mean,
    "median": np.median,
    "mode": lambda x: stats.mode(x)[0],
}


def validate_feature_group_dtypes(func: Callable) -> Callable:
    """A decorator to validate data types of feature groups in the dataset.

    Raises a ValueError if any feature group contains columns of different data types.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.

    """

    def wrapper(self: "AggFunc", *args: Any, **kwargs: Any) -> Any:
        func(self, *args, **kwargs)
        transformed_data = self.dataset.data.copy()
        feature_groups = self.dataset.feature_groups(names=True)

        for feature_group in feature_groups:
            dtypes = transformed_data[feature_group].dtypes
            unique_dtypes = dtypes.unique()

            if len(unique_dtypes) > 1:
                raise ValueError(
                    f"Inconsistent dtypes within feature group {feature_group}. "
                    f"Expected a single dtype, but found {unique_dtypes}."
                )

    return wrapper


def validate_aggregation_func(func: Callable) -> Callable:
    """A decorator to validate the aggregation function passed to the class.

    Raises a ValueError if the aggregation function is not valid. Valid functions are those in AGG_FUNCS or any
    callable function.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.

    """

    def wrapper(self: "AggFunc", *args: Any, **kwargs: Any) -> Any:
        aggregation_func = kwargs.get("aggregation_func")
        if aggregation_func is None and args:  # pragma: no cover
            aggregation_func = args[0]
        if isinstance(aggregation_func, str):
            valid_funcs = list(AGG_FUNCS.keys())
            if aggregation_func not in valid_funcs:
                raise ValueError(f"Invalid aggregation function: {aggregation_func}. Choose from {valid_funcs}.")
        elif not callable(aggregation_func):
            raise ValueError(
                f"aggregation_func must be either a string (one of {list(AGG_FUNCS.keys())}) or a function."
            )
        return func(self, *args, **kwargs)

    return wrapper


def init_ray(func: Callable) -> Callable:  # pragma: no cover
    """A decorator to initialize the Ray library for parallel processing if it's not already initialised.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.

    """

    def wrapper(self: "AggFunc", *args: Any, **kwargs: Any) -> Any:
        func(self, *args, **kwargs)
        if self.parallel and not ray.is_initialized():
            if self.num_cpus != -1:
                ray.init(num_cpus=self.num_cpus)
            else:
                ray.init()

    return wrapper


def get_agg_feature(
    data: pd.DataFrame, feature_group: List[str], agg_func: Callable, agg_func_name: str
) -> pd.DataFrame:
    """Apply the aggregation function to the feature group and return a DataFrame with the aggregated feature.

    Args:
        data (pd.DataFrame): The DataFrame to aggregate.
        feature_group (List[str]): The list of column names that form the feature group.
        agg_func (Callable): The aggregation function to apply.
        agg_func_name (str): The name of the aggregation function.

    Returns:
        pd.DataFrame: A DataFrame with a single column, which is the aggregated feature.

    """
    agg_feature = data[feature_group].agg(agg_func, axis=1)
    name = re.sub(r"_w\d+$", "", feature_group[0])
    feature_name = f"agg_{agg_func_name}_{name}"
    return pd.DataFrame({feature_name: agg_feature})


@ray.remote
def _aggregate(
    feature_group: List[str], data: pd.DataFrame, agg_func: Callable, aggregation_func_name: str
) -> pd.DataFrame:  # pragma: no cover
    """Remote function to apply the aggregation function to the feature group using Ray for parallel processing.

    This function is decorated with @ray.remote to allow it to be executed in a distributed manner by Ray. If the
    aggregation function is "mean" or "median" but the feature group is categorical, a warning is issued and the
    aggregation function is changed to "mode".

    Args:
        feature_group (List[str]): The list of column names that form the feature group.
        data (pd.DataFrame): The DataFrame to aggregate.
        agg_func (Callable): The aggregation function to apply.
        aggregation_func_name (str): The name of the aggregation function.

    Returns:
        pd.DataFrame: A DataFrame with a single column, which is the aggregated feature.

    """
    if (
        aggregation_func_name not in ["mean", "median"]
        or not (data[feature_group].dtypes.apply(lambda x: x == "object")).all()
    ):
        return get_agg_feature(data, feature_group, agg_func, aggregation_func_name)
    warnings.warn(
        f"Aggregation function is {aggregation_func_name} but feature group {feature_group} is "
        "categorical. Using mode instead."
    )

    def agg_mode_func(x):
        return stats.mode(x)[0]

    return get_agg_feature(data, feature_group, agg_mode_func, "mode")


# pylint: disable=R0903
class AggFunc:
    """Class for Aggregation Function method of data transformation in longitudinal data analysis.

    This class provides functionality for applying an aggregation function to feature groups in a longitudinal dataset.
    A feature group is defined as a group of features that share the same base name but come from different waves of
    data collection. For example, if a dataset contains the features "income_wave1", "income_wave2", and "income_wave3",
    these features form a group.

    The aggregation function is applied across the waves for each feature group, resulting in a single aggregated
    feature for each group. For example, if the aggregation function is the mean, then the "income_wave1",
    "income_wave2", and "income_wave3" features would be replaced with a single "mean_income" feature.

    The class now supports custom aggregation functions provided they are callable. You can pass a function that takes
    a pandas Series and returns a single value.

    The class also supports parallel processing for the aggregation using the Ray library. This can significantly
    speed up the transformation for large datasets.

    Attributes:
    -----------
    dataset: LongitudinalDataset
        The longitudinal dataset to apply the transformation to.
    aggregation_func: str or Callable
        The aggregation function to apply to the feature groups. Can be "mean", "median", "mode", or a custom function.
    parallel: bool
        Whether to use parallel processing for the aggregation.
    num_cpus: int
        The number of CPUs to use for parallel processing. Defaults to -1, which uses all available CPUs.

    Example:
    --------
    ```python
    # Create a LongitudinalDataset object
    dataset = LongitudinalDataset(data)

    # Initialize the AggFunc object with "mean" as the aggregation function
    agg_func = AggFunc(dataset, aggregation_func="mean")

    # Initialize the AggFunc object with a custom aggregation function
    custom_func = lambda x: x.quantile(0.25) # returns the first quartile
    agg_func = AggFunc(dataset, aggregation_func=custom_func)

    # Apply the transformation
    transformed_dataset = agg_func.transform()
    ```

    """

    @validate_feature_group_dtypes
    @validate_aggregation_func
    @init_ray
    def __init__(
        self,
        dataset: LongitudinalDataset,
        aggregation_func: Union[str, Callable] = "mean",
        parallel: bool = False,
        num_cpus: int = -1,
    ):
        self.dataset = dataset
        self.aggregation_func = aggregation_func
        self.parallel = parallel
        self.num_cpus = num_cpus
        if isinstance(aggregation_func, str):
            self.agg_func = AGG_FUNCS[aggregation_func]
        else:
            self.agg_func = aggregation_func

    def transform(self):
        """Apply the aggregation function to the feature groups in the dataset.

        This method applies the aggregation function to each feature group in the dataset, replacing the group with
        a single aggregated feature. If parallel processing is enabled, the aggregation is performed in parallel
        using Ray.

        Warning: If the aggregation function is "mean" or "median" and the feature group is categorical data, the
        aggregation function will be changed to "mode" and a warning will be displayed.

        Returns:
        --------
        pandas.DataFrame
            The transformed dataset, where each feature group has been replaced with a single aggregated feature.

        """
        transformed_data = self.dataset.data.copy()
        feature_groups = self.dataset.feature_groups(names=True)

        if self.parallel:
            non_grouped_data = transformed_data[self.dataset.non_longitudinal_features(names=True)]
            tasks = [
                _aggregate.remote(feature_group, transformed_data, self.agg_func, self.aggregation_func)
                for feature_group in feature_groups
            ]
            results = ray.get(tasks)
            transformed_data = pd.concat([non_grouped_data, pd.concat(results, axis=1)], axis=1)
        else:
            for feature_group in feature_groups:
                if (
                    self.aggregation_func in ["mean", "median"]
                    and (transformed_data[feature_group].dtypes.apply(lambda x: x == "object")).all()
                ):
                    warnings.warn(
                        f"Aggregation function is {self.aggregation_func} but feature group {feature_group} is "
                        "categorical. Using mode instead."
                    )

                    def agg_mode_func(x):
                        return stats.mode(x)[0]

                    agg_feature_df = get_agg_feature(transformed_data, feature_group, agg_mode_func, "mode")
                else:
                    agg_feature_df = get_agg_feature(
                        transformed_data, feature_group, self.agg_func, self.aggregation_func
                    )

                transformed_data = pd.concat([transformed_data, agg_feature_df], axis=1)
                transformed_data.drop(feature_group, axis=1, inplace=True)

        self.dataset.set_data(transformed_data)
        self.dataset._feature_groups = None  # pylint: disable=W0212
        return self.dataset


def main():  # pragma: no cover
    longitudinal_data = LongitudinalDataset("./data/elsa/core/csv/dementia_dataset.csv")
    longitudinal_data.load_data_target_train_test_split(
        target_column="class_dementia_w8",
        remove_target_waves=True,
    )
    longitudinal_data.setup_features_group(input_data="elsa")

    aggr_func = AggFunc(longitudinal_data, aggregation_func="mean", parallel=False)
    transformed_data_parallel_false = aggr_func.transform()
    print(transformed_data_parallel_false.data.head())


if __name__ == "__main__":  # pragma: no cover
    main()
