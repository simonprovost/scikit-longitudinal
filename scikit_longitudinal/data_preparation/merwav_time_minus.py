# pylint: disable=R0801

from typing import List, Union

import numpy as np
from overrides import override

from scikit_longitudinal.templates.custom_data_preparation_mixin import DataPreparationMixin


class MerWavTimeMinus(DataPreparationMixin):
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
        return {}

    @override
    def _prepare_data(self, X: np.ndarray, y: np.ndarray = None) -> "MerWavTimeMinus":
        """Prepare the data for the transformation.

        This method is overridden from the DataPreparationMixin class. It takes numpy arrays as input and converts them
        into pandas DataFrame and numpy array for further processing according to the class's description above.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The target data. Not particularly relevant for this class. We store the target but do not
                use it for the transformation.

        Returns:
            MerWavTimeMinus: The instance of the class with prepared data.

        """
        return self
