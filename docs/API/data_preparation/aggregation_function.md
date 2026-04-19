# Aggregation Function

??? tip "What is the AggrFunc module?"
    The `AggrFunc` module facilitates the application of aggregation functions to feature groups within a longitudinal
    dataset, enabling the use of temporal information before applying traditional machine learning algorithms.

??? question "What are features_group and non_longitudinal_features?"
    Two key attributes, `features_group` and `non_longitudinal_features`, enable algorithms to interpret the
    temporal structure of longitudinal data.

    - **features_group**: A list of lists where each sublist contains indices of a longitudinal attribute's
      waves, ordered from oldest to most recent. This captures temporal dependencies.
    - **non_longitudinal_features**: A list of indices for static, non-temporal features excluded from the
      temporal matrix.

    Proper setup of these attributes is critical for leveraging temporal patterns effectively.

    [See More In Temporal Dependency Guide :fontawesome-solid-timeline:](../../tutorials/temporal_dependency.md){ .md-button }

## ::: scikit_longitudinal.data_preparation.aggregation_function.AggrFunc
    options:
        heading: "AggrFunc"
        members:
            - get_params
            - _prepare_data
            - _transform
