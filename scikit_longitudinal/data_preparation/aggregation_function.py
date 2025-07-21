# pylint: disable=R0801

import re
import warnings
from typing import Any, Callable, List, Union

import numpy as np
import pandas as pd
import ray
from overrides import override
from scipy import stats

from scikit_longitudinal.data_preparation.longitudinal_dataset import clean_padding
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

    The decorator performs a validation check on the indices of the features group. This check ensures that the
    indices fall within the valid range, which is defined as being within the total number of features in the dataset.
    In the event that the specified condition is not met, a ValueError exception is raised.

    Args:
        func (Callable):
            The function to be decorated.

    Returns:
        Callable:
            The decorated function.

    """

    def wrapper(self: "AggrFunc", *args: Any, **kwargs: Any) -> Any:
        features_group = self.features_group
        if not features_group:
            raise ValueError("features_group cannot be None.")
        if features_group:
            for feature_group in features_group:
                if any(index >= self.dataset.shape[1] for index in feature_group):
                    raise ValueError(
                        f"Invalid index within feature group {feature_group}. "
                        "All indices should be within the range of the feature set."
                    )
        return func(self, *args, **kwargs)

    return wrapper


def validate_aggregation_func(func: Callable) -> Callable:
    """A decorator to validate the aggregation function passed to the class.

    The decorator performs a validation check on the provided aggregation function to ensure its validity. The set of
    valid functions consists of all functions present in the AGG_FUNCS collection, as well as any other functions
    that are callable. In the event that the specified condition is not met, a ValueError will be raised.

    Args:
        func (Callable):
            The function to be decorated.

    Returns:
        Callable:
            The decorated function.

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
    """A decorator to initialise the Ray library for parallel processing.

    The purpose of this decorator is to verify the initialisation status of the Ray framework. Init if it is not
    already - Furthermore, if the specified number of CPUs is not provided, the system will proceed to initialise Ray
    by utilising all available CPUs.

    Args:
        func (Callable):
            The function to be decorated.

    Returns:
        Callable:
            The decorated function.

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

    The provided function is designed to execute the specified aggregation function on the feature group within the
    input DataFrame. The result of this operation is a newly generated DataFrame that contains the aggregated feature
    while removing the original longitudinal features.

    Args:
        data (pd.DataFrame):
            The DataFrame to aggregate.
        feature_group (List[str]):
            The list of column names that form the feature group.
        agg_func (Callable):
            The aggregation function to apply.
        agg_func_name (str):
            The name of the aggregation function.

    Returns:
        pd.DataFrame:
            A DataFrame with a single column, which is the aggregated feature.

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

    The present function has been decorated with the @ray.remote decorator, which serves the purpose of facilitating
    distributed execution through the utilisation of Ray. Note, however, when the aggregation function is specified as
    either "mean" or "median" and the feature group is categorical, the system automatically switches the aggregation
    function to "mode" and generates a warning message.

    Args:
        feature_group (List[str]):
            The list of column names that form the feature group.
        data (pd.DataFrame):
            The DataFrame to aggregate.
        agg_func (Callable):
            The aggregation function to apply.
        aggregation_func_name (str):
            The name of the aggregation function.

    Returns:
        pd.DataFrame:
            A DataFrame with a single column, which is the aggregated feature.

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
    """AggrFunc stands for Aggregation Functions, aggregation on feature groups in longitudinal datasets.

    The `AggrFunc` facilitates the application of aggregation functions to feature groups within a longitudinal
    dataset, enabling the use of `temporal information` before applying traditional machine learning algorithms like
    those in Scikit-Learn or any other alike machine learning-based libarires.


    !!! question "What is a feature group?"
        In a nutshell, a feature group is a collection of features sharing a common base longitudinal attribute
        across different waves of data collection (e.g., "income_wave1", "income_wave2", "income_wave3"). Note that
        aggregation reduces the dataset's temporal information significantly.

        To see more, we highly recommend visiting the `Temporal Dependency` page in the documentation.

        [Temporal Dependency Guide :fontawesome-solid-timeline:](https://scikit-longitudinal.readthedocs.io/latest/tutorials/temporal_dependency/){ .md-button }


    The aggregation function is applied iteratively across waves for each feature group, producing a single aggregated
    feature per group (e.g., "mean_income" from "income_wave1", "income_wave2", "income_wave3" using the "mean"
    function). Supported aggregation functions include "mean", "median", "mode", and custom callable functions that
    take a pandas Series as input and return a single value. Parallel processing is also supported via the Ray library
    for enhanced efficiency on large datasets.

    Args:
        features_group (List[List[int]], optional): A temporal matrix representing the temporal dependency of a
            longitudinal dataset. Each sublist contains indices of a longitudinal attribute's waves. Defaults to None.
            See the "Temporal Dependency" page in the documentation for details.
        non_longitudinal_features (List[Union[int, str]], optional): A list of indices or names of non-longitudinal
            features. Defaults to None.
        feature_list_names (List[str], optional): A list of feature names in the dataset. Defaults to None.
        aggregation_func (Union[str, Callable], optional): The aggregation function to apply. Options are "mean",
            "median", "mode", or a custom callable function. Defaults to "mean". See further in
            the `aggregation_function.py` file at the object `AGG_FUNCS` for those supported.
        parallel (bool, optional): Whether to use parallel processing for aggregation. Defaults to False.
        num_cpus (int, optional): Number of CPUs for parallel processing. Defaults to -1 (uses all available CPUs).

    Attributes:
        dataset (pd.DataFrame): The longitudinal dataset to transform.
        aggregation_func (Union[str, Callable]): The aggregation function applied to feature groups.
        parallel (bool): Whether parallel processing is enabled.
        num_cpus (int): Number of CPUs used for parallel processing.

    Examples:
        Below are examples demonstrating the usage of the `AggrFunc` class with the "stroke.csv" dataset.
        Please, note that "stroke.csv" is a placeholder and should be replaced with the actual path to your dataset.

        !!! example "Basic Usage with Mean Aggregation"
            ```python
            from scikit_longitudinal.data_preparation import LongitudinalDataset
            from scikit_longitudinal.data_preparation.aggregation_function import AggrFunc

            # Load dataset
            dataset = LongitudinalDataset('./stroke_longitudinal.csv')
            dataset.load_data()
            dataset.load_target(target_column="stroke_w2")
            dataset.setup_features_group("elsa")
            dataset.load_train_test_split(test_size=0.2, random_state=42)

            # Initialize AggrFunc
            agg_func = AggrFunc(
                aggregation_func="mean",
                features_group=dataset.feature_groups(),
                non_longitudinal_features=dataset.non_longitudinal_features(),
                feature_list_names=dataset.data.columns.tolist()
            )

            # Apply transformation
            agg_func.prepare_data(dataset.X_train)
            transformed_dataset, _, _, _ = agg_func._transform()
            ```

        !!! example "Using Custom Aggregation Function"
            ```python
            from scikit_longitudinal.data_preparation import LongitudinalDataset
            from scikit_longitudinal.data_preparation.aggregation_function import AggrFunc

            # Load dataset
            dataset = LongitudinalDataset("./stroke_longitudinal.csv")
            dataset.load_data()
            dataset.load_target(target_column="stroke_w2")
            dataset.setup_features_group("elsa")
            dataset.load_train_test_split(test_size=0.2, random_state=42)

            # Define custom function
            custom_func = lambda x: x.quantile(0.25)  # First quartile

            # Initialize AggrFunc
            agg_func = AggrFunc(
                aggregation_func=custom_func,
                features_group=dataset.feature_groups(),
                non_longitudinal_features=dataset.non_longitudinal_features(),
                feature_list_names=dataset.data.columns.tolist(),
            )

            # Apply transformation
            agg_func.prepare_data(dataset.X_train)
            transformed_dataset, _, _, _ = agg_func._transform()
            ```

        !!! example "Using Parallel Processing"
            ```python
            # ... similar to the previous example, prepare data and transform ...

            # Initialize AggrFunc with parallel processing
            agg_func = AggrFunc(
                aggregation_func="mean",
                features_group=dataset.feature_groups(),
                non_longitudinal_features=dataset.non_longitudinal_features(),
                feature_list_names=dataset.data.columns.tolist(),
                parallel=True, # Enable parallel processing
                num_cpus=4 # Specify number of CPUs (optional, -1 for all available)
            )

            # ... similar to the previous example, prepare data and transform ...
            ```
    """

    @validate_aggregation_func
    @init_ray
    def __init__(
        self,
        features_group: List[List[int]] = None,
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

    def get_params(self, deep: bool = True):  # pylint: disable=W0613
        """Get the parameters of the AggrFunc instance.

        This method retrieves the configuration parameters of the `AggrFunc` instance, useful for inspection or
        hyperparameter tuning.

        Args:
            deep (bool, optional): Unused parameter but kept for consistency with the scikit-learn API.

        Returns:
            dict: The parameters of the AggrFunc instance.
        """
        return {
            "aggregation_func": self.aggregation_func,
            "parallel": self.parallel,
            "num_cpus": self.num_cpus,
        }

    @override
    def _prepare_data(self, X: np.ndarray, y: np.ndarray = None) -> "AggrFunc":
        """Prepare the data for transformation.

        This method, overridden from `DataPreparationMixin`, converts input numpy arrays into a pandas DataFrame and
        stores the target data for compatibility, though the target is not used in the transformation.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray, optional): The target data. Defaults to None.

        Returns:
            AggrFunc: The instance with prepared data.
        """
        self.dataset = pd.DataFrame(X, columns=self.feature_list_names)
        self.target = y

        return self

    @validate_feature_group_indices
    def _transform(self):
        """Apply the aggregation function to feature groups in the dataset.

        This method applies the specified aggregation function to each feature group, replacing it with a single
        aggregated feature.

        !!! tip "Parallel Processing"
            If parallel processing is enabled, it uses the Ray library.

        !!! note "Categorical Data Handling"
            For "mean" or "median" functions
            with categorical data, it switches to "mode" and issues a warning automatically.

        Returns:
            tuple:
                - [x] pd.DataFrame: The transformed dataset.
                - [x] List[List[int]]: Feature groups in the transformed dataset (None, as they are aggregated).
                - [x] List[Union[int, str]]: Non-longitudinal features in the transformed dataset (None).
                - [x] List[str]: Names of features in the transformed dataset.
        """
        if self.features_group is not None:
            self.features_group = clean_padding(self.features_group)

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
