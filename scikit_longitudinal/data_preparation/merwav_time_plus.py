# pylint: disable=R0801

from typing import List, Union

import numpy as np
from overrides import override

from scikit_longitudinal.templates.custom_data_preparation_mixin import DataPreparationMixin


class MerWavTimePlus(DataPreparationMixin):
    """MerWavTimePlus stands for Merge waves while keeping time indices in longitudinal datasets.

    The `MerWavTimePlus` class transforms longitudinal data by merging all features across waves into a single set
    while preserving their time indices. This maintains the temporal structure, enabling longitudinal machine learning
    methods to leverage temporal dependencies and patterns.

    !!! quote "MerWavTime(+)? Usefulness?"
        In longitudinal studies, data is collected across multiple waves (time points), resulting in features that
        capture temporal information. This method merges all features from all waves into a single set while preserving
        their time indices, facilitating the use of time-aware machine learning techniques.

    !!! question "What is a feature group?"
        In a nutshell, a feature group is a collection of features sharing a common base longitudinal attribute
        across different waves of data collection (e.g., "income_wave1", "income_wave2", "income_wave3"). Note that
        aggregation reduces the dataset's temporal information significantly.

        To see more, we highly recommend visiting the `Temporal Dependency` page in the documentation.

        [Temporal Dependency Guide :fontawesome-solid-timeline:](https://scikit-longitudinal.readthedocs.io/latest/tutorials/temporal_dependency/){ .md-button }


    Args:
        features_group (List[List[int]], optional): A temporal matrix representing the temporal dependency of a
            longitudinal dataset. Each sublist contains indices of a longitudinal attribute's waves. Defaults to None.
        non_longitudinal_features (List[Union[int, str]], optional): A list of indices or names of non-longitudinal
            features. Defaults to None.
        feature_list_names (List[str], optional): A list of feature names in the dataset. Defaults to None.

    Attributes:
        features_group (List[List[int]]): The temporal matrix of feature groups.
        non_longitudinal_features (List[Union[int, str]]): The non-longitudinal features.
        feature_list_names (List[str]): The feature names in the dataset.

    Examples:
        Below is an example using the "stroke.csv" dataset to demonstrate the `MerWavTimePlus` class.
        Please, note that "stroke.csv" is a placeholder and should be replaced with the actual path to your dataset.

        !!! example "Basic Usage"
            ```python
            from scikit_longitudinal.data_preparation import LongitudinalDataset
            from scikit_longitudinal.data_preparation import MerWavTimePlus

            # Load dataset
            dataset = LongitudinalDataset('./stroke_longitudinal.csv')
            dataset.load_data()
            dataset.load_target(target_column="stroke_w2")
            dataset.setup_features_group("elsa")
            dataset.load_train_test_split(test_size=0.2, random_state=42)

            # Initialize MerWavTimePlus
            mer_wav_plus = MerWavTimePlus(
                features_group=dataset.feature_groups(),
                non_longitudinal_features=dataset.non_longitudinal_features(),
                feature_list_names=dataset.data.columns.tolist()
            )

            # No need to apply any transformation, MerWavTimePlus takes the dataset as it is
            # Meaning that it keeps the temporal dependency intact.

            # Later on, primitives understand this temporal dependency via the `features_group` attribute.
            ```
    """
    def __init__(
        self,
        features_group: List[List[int]] = None,
        non_longitudinal_features: List[Union[int, str]] = None,
        feature_list_names: List[str] = None,
    ):
        self.features_group = features_group
        self.non_longitudinal_features = non_longitudinal_features
        self.feature_list_names = feature_list_names

    def get_params(self, deep: bool = True):  # pylint: disable=W0613
        """Get the parameters of the MerWavTimePlus instance.

        Retrieves the configuration parameters of the instance, useful for inspection or integration with scikit-learn
        pipelines.

        Args:
            deep (bool, optional): Unused parameter but kept for consistency with the scikit-learn API.

        Returns:
            dict: The parameters of the MerWavTimePlus instance.
        """
        return {}

    @override
    def _prepare_data(self, X: np.ndarray, y: np.ndarray = None) -> "MerWavTimePlus":
        """Prepare the data for transformation.

        Overridden from `DataPreparationMixin`.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray, optional): The target data, stored but not used in transformation. Defaults to None.

        Returns:
            MerWavTimePlus: The instance with prepared data.
        """
        return self
