# Longitudinal Dataset

??? tip "What is the LongitudinalDataset module?"
    The `LongitudinalDataset` module is a comprehensive container designed for managing and preparing longitudinal datasets.
    It provides essential data management and transformation capabilities, facilitating the development and application
    of machine learning algorithms tailored to longitudinal data classification tasks. Built around a `pandas` DataFrame,
    it enhances functionality while maintaining a familiar interface.

    We highly recommend reviewing the `Temporal Dependency` page in the documentation for a deeper understanding
    of feature groups and the `LongitudinalDataset` module's usage before exploring its API.

    [See the Temporal Dependency Guide](../../tutorials/temporal_dependency.md){ .md-button }

## ::: scikit_longitudinal.data_preparation.longitudinal_dataset.LongitudinalDataset
    options:
        heading: "LongitudinalDataset"
        members:
            - load_data
            - load_target
            - load_train_test_split
            - setup_features_group
            - feature_groups
            - non_longitudinal_features
            - convert
            - save_data
            - set_data
            - set_target
            - setX_train
            - setX_test
            - sety_train
            - sety_test
