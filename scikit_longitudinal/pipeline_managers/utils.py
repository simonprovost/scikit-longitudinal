from typing import Callable, List, Optional, Union

from sklearn.base import TransformerMixin

from scikit_longitudinal.templates import DataPreparationMixin


def configure_transformer(
    transformer: TransformerMixin,
    name: str,
    features_group: List[List[int]],
    non_longitudinal_features: List[Union[int, str]],
    feature_list_names: List[str],
    update_feature_groups_callback: Optional[Callable] = None,
) -> TransformerMixin:
    """Configures the transformer with the necessary attributes based on its type.

    If the transformer is of type `DataPreparationMixin`, it is configured with the necessary
    attributes for data preparation. If it has the attribute 'features_group', it is configured
    as a feature selector for longitudinal data. Otherwise, it's assumed to be a standard
    scikit-learn transformer.

    Args:
        transformer (TransformerMixin):
            The transformer to be configured.
        name (str):
            Name of the transformer.
        features_group (List[List[int]]):
            Grouping of the Longitudinal features.
        non_longitudinal_features (List[Union[int, str]]):
            List of non-longitudinal features.
        feature_list_names (List[str]):
            List of names corresponding to the features.
        update_feature_groups_callback (Optional[Callable]):
            Callback function to update feature groups. It must be a callable if provided.

    Returns:
        TransformerMixin:
            The configured transformer.

    Raises:
        ValueError:
            - If required attributes are missing for the transformer type.
            - If the callback function for updating feature groups is not provided for transformers
              that require it.

    """

    def check_required_attributes():
        """
        Check if the required attributes are set.
        """
        if not features_group:
            raise ValueError(f"The transformer {name} requires 'features_group'.")
        if not feature_list_names:
            raise ValueError(
                f"The features_group attribute of the transformer named {name} designates it as a "
                "longitudinal-based transformer. But no feature_list_names were provided. "
                "Please supply the names of the features."
            )

    def check_update_feature_callback():
        """
        Check if the update_feature_groups_callback is valid.
        """
        if update_feature_groups_callback is None:
            raise ValueError(
                f"The transformer {name} has a features_group attribute, but no update_data"
                " callback was provided. Ensure an update_data_callback function is passed to"
                " update not only the selected features but also the non-longitudinal features."
            )
        if not callable(update_feature_groups_callback):
            raise ValueError("update_data_callback must be a callable function")

    def set_transformer_attributes():
        """
        Set the required attributes for the transformer.
        """
        transformer.features_group = features_group
        if hasattr(transformer, "non_longitudinal_features") and non_longitudinal_features:
            transformer.non_longitudinal_features = non_longitudinal_features

    if isinstance(transformer, DataPreparationMixin):
        check_required_attributes()
        set_transformer_attributes()
        transformer.feature_list_names = feature_list_names

    elif hasattr(transformer, "features_group"):  # For exemple, this could be Feature Selection for Longitudinal data
        check_required_attributes()
        check_update_feature_callback()
        set_transformer_attributes()

    # Else could simply be standard scikit-learn transformer
    return transformer
