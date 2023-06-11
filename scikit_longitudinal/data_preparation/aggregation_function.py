import re
import warnings
from typing import Any, Callable, List, Union

import numpy as np
import pandas as pd
import ray
from overrides import override
from scipy import stats

from scikit_longitudinal.templates.custom_data_preparation_mixin import DataPreparationMixin

def mode_with_keepdims(x, keepdims=True):
    return stats.mode(x, keepdims=keepdims)[0][0]

AGG_FUNCS = {
    "mean": np.mean,
    "median": np.median,
    "mode": mode_with_keepdims,
}


def validate_feature_group_indices(func: Callable) -> Callable:
    """A decorator to validate feature group indices.

    The decorator checks if the indices of the features group are within the valid range, i.e., they should be within
    the total number of features in the dataset and non-negative. If not, it raises a ValueError.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.

    """

    def wrapper(self: "AggrFunc", *args: Any, **kwargs: Any) -> Any:
        features_group = self.features_group
        if not features_group:
            raise ValueError("features_group cannot be None.")
        if features_group:
            for feature_group in features_group:
                if any(index >= self.dataset.shape[1] or index < 0 for index in feature_group):
                    raise ValueError(
                        f"Invalid index within feature group {feature_group}. "
                        "All indices should be within the range of the feature set."
                    )
        return func(self, *args, **kwargs)

    return wrapper


def validate_aggregation_func(func: Callable) -> Callable:
    """A decorator to validate the aggregation function passed to the class.

    The decorator checks if the aggregation function is a valid one. Valid functions are those in AGG_FUNCS or any
    other callable function. If not, it raises a ValueError.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.
    """

    def wrapper(self: "AggrFunc", *args: Any, **kwargs: Any) -> Any:
        aggregation_func = kwargs.get("aggregation_func")
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


def init_ray(func: Callable) -> Callable:
    """A decorator to initialize the Ray library for parallel processing if it's not already initialised.

    This decorator checks if Ray is initialized. If not, it initializes Ray using the specified number of CPUs or
    all available CPUs if the number is not specified.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.
    """

    def wrapper(self: "AggrFunc", *args: Any, **kwargs: Any) -> Any:
        parallel = kwargs.get("parallel")
        num_cpus = kwargs.get("num_cpus")
        if parallel and not ray.is_initialized():
            if num_cpus != -1:
                ray.init(num_cpus=num_cpus)
            else:
                ray.init()
        return func(self, *args, **kwargs)

    return wrapper


def get_agg_feature(
    data: pd.DataFrame, feature_group: List[str], agg_func: Callable, agg_func_name: str
) -> pd.DataFrame:
    """Apply the aggregation function to the feature group and return a DataFrame with the aggregated feature.

    This function applies the given aggregation function to the feature group in the input DataFrame and generates a
    new DataFrame with the aggregated feature.

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

    This function is decorated with @ray.remote to enable distributed execution by Ray. If the aggregation function
    is "mean" or "median" but the feature group is categorical, it switches to "mode" as the aggregation function and
    issues a warning.

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
class AggrFunc(DataPreparationMixin):
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
        dataset: LongitudinalDataset
            The longitudinal dataset to apply the transformation to.
        aggregation_func: str or Callable
            The aggregation function to apply to the feature groups. Can be "mean", "median", "mode", or a custom
            function. If providing a custom function, it should be a function that takes a pandas Series and returns
            a single value. If the function is not callable, a ValueError will be raised during validation. If the
            aggregation function is "mean" or "median" and the feature group is categorical, the function is switched
            to "mode" and a warning is issued.
        parallel: bool
            Whether to use parallel processing for the aggregation.
        num_cpus: int
            The number of CPUs to use for parallel processing. Defaults to -1, which uses all available CPUs.

    Example:
        ```python
        # Create a LongitudinalDataset object
        dataset = LongitudinalDataset(data)

        # Initialize the AggrFunc object with "mean" as the aggregation function
        agg_func = AggrFunc(
            aggregation_func="mean",
            features_group=dataset.feature_groups(),
            non_longitudinal_features=dataset.non_longitudinal_features(),
            feature_list_names=dataset.data.columns.tolist()
        )

        # Initialize the AggrFunc object with a custom aggregation function
        custom_func = lambda x: x.quantile(0.25) # returns the first quartile
        agg_func = AggrFunc(
            aggregation_func=custom_func
            features_group=dataset.feature_groups(),
            non_longitudinal_features=dataset.non_longitudinal_features(),
            feature_list_names=dataset.data.columns.tolist()
        ),

        # Apply the transformation
        agg_func.prepare_data(dataset.data)
        transformed_dataset, transformed_features_group, transformed_non_longitudinal_features, \
            transformed_feature_list_names = agg_func.transform()
        ```
    """

    @validate_aggregation_func
    @init_ray
    def __init__(
        self,
        features_group: List[List[Union[int, str]]] = None,
        non_longitudinal_features: List[Union[int, str]] = None,
        feature_list_names: List[str] = None,
        aggregation_func: Union[str, Callable] = "mean",
        parallel: bool = False,
        num_cpus: int = -1,
    ):
        self.dataset = pd.DataFrame([])
        self.target = np.ndarray([])
        self.features_group = features_group
        self.non_longitudinal_features = non_longitudinal_features
        self.feature_list_names = feature_list_names
        self.aggregation_func = aggregation_func
        self.parallel = parallel
        self.num_cpus = num_cpus
        if isinstance(aggregation_func, str):
            self.agg_func = AGG_FUNCS[aggregation_func]
        else:
            self.agg_func = aggregation_func

    @override
    def _prepare_data(self, X: np.ndarray, y: np.ndarray = None) -> "AggrFunc":
        """Prepare the data for the transformation.

        This method is overridden from the DataPreparationMixin class. It takes numpy arrays as input and converts them
        into pandas DataFrame and numpy array for further processing according to the class's description above.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The target data. Not particularly relevant for this class. We store the target but do not
                use it for the transformation.

        Returns:
            AggrFunc: The instance of the class with prepared data.

        """
        self.dataset = pd.DataFrame(X, columns=self.feature_list_names)
        self.target = y

        return self

    @validate_feature_group_indices
    def _transform(self):
        """Apply the aggregation function to the feature groups in the dataset.

        This method applies the aggregation function to each feature group in the dataset, replacing the group with
        a single aggregated feature. If parallel processing is enabled, the aggregation is performed in parallel
        using Ray.

        If the aggregation function is "mean" or "median" and the feature group is categorical data, the aggregation
        function is switched to "mode" and a warning is displayed.

        Returns:
            pd.DataFrame: The transformed dataset.
            List[List[Union[int, str]]]: The feature groups in the transformed dataset.
            List[Union[int, str]]: The non-longitudinal features in the transformed dataset.
            List[str]: The names of the features in the transformed dataset.
        """
        transformed_data = self.dataset.copy()
        feature_groups = [transformed_data.columns[i].tolist() for i in self.features_group]

        if self.parallel:
            non_grouped_data = transformed_data.iloc[:, self.non_longitudinal_features]
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

        self.dataset = transformed_data
        self.features_group = None  # pylint: disable=W0212
        self.non_longitudinal_features = None  # pylint: disable=W0212
        self.feature_list_names = self.dataset.columns.tolist()
        return self.dataset, self.features_group, self.non_longitudinal_features, self.feature_list_names
