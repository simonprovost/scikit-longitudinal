# pylint: disable=R0801

from typing import List, Union

import numpy as np
from overrides import override

from scikit_longitudinal.templates.custom_data_preparation_mixin import (
    DataPreparationMixin,
)


class MerWavTimeMinus(DataPreparationMixin):
    """MerwavTimeMinus stands for Merge Waves yet discards time indices in longitudinal datasets.

    The `MerWavTimeMinus` transforms longitudinal data by merging all features across waves into a single set,
    effectively discarding temporal information. This approach treats different values of the same original longitudinal
    feature as distinct features, simplifying the dataset for traditional machine learning algorithms but losing temporal
    dependencies.

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
        Below is an example demonstrating the usage of the `MerWavTimeMinus` class with the "stroke.csv" dataset.
        Please, note that "stroke.csv" is a placeholder and should be replaced with the actual path to your dataset.

        !!! example "Basic Usage"
            ```python
            from scikit_longitudinal.data_preparation import MerWavTimeMinus

            # Load dataset
            dataset = LongitudinalDataset('./stroke_longitudinal.csv')
            dataset.load_data()
            dataset.load_target(target_column="stroke_w2")
            dataset.setup_features_group("elsa")
            dataset.load_train_test_split(test_size=0.2, random_state=42)

            # Initialize MerWavTimeMinus
            mer_wav = MerWavTimeMinus(
                features_group=dataset.feature_groups(),
                non_longitudinal_features=dataset.non_longitudinal_features(),
                feature_list_names=dataset.data.columns.tolist()
            )

            # No need to apply any transformation, MerWavTimeMinus takes the dataset as it is
            # Meaning that it does not care about any of the temporal dependency.

            # We let this there for compatibility but it has little value alone.
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
        """Get the parameters of the MerWavTimeMinus instance.

        This method retrieves the configuration parameters of the `MerWavTimeMinus` instance, useful for inspection or
        hyperparameter tuning.

        Args:
            deep (bool, optional): Unused parameter but kept for consistency with the scikit-learn API.

        Returns:
            dict: The parameters of the MerWavTimeMinus instance.
        """
        return {}

    @override
    def _prepare_data(self, X: np.ndarray, y: np.ndarray = None) -> "MerWavTimeMinus":
        """Prepare the data for transformation.

        Overridden from `DataPreparationMixin`.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray, optional): The target data. Defaults to None.

        Returns:
            MerWavTimeMinus: The instance with prepared data.
        """
        return self
