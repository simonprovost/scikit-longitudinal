# Merging Waves and Discarding Time Indices

??? tip "What is the MerWavTimeMinus module?"
    The `MerWavTimeMinus` module transforms longitudinal data by merging all features across waves into a single set,
    discarding temporal information. This simplifies the dataset for traditional machine learning algorithms but loses
    temporal dependencies. It provides methods for data preparation and transformation, including `prepare_data` and
    `transform`.

??? question "What are features_group and non_longitudinal_features?"
    Two key attributes, `features_group` and `non_longitudinal_features`, enable algorithms to interpret the
    temporal structure of longitudinal data.

    - **features_group**: A list of lists where each sublist contains indices of a longitudinal attribute's
      waves, ordered from oldest to most recent. This captures temporal dependencies.
    - **non_longitudinal_features**: A list of indices for static, non-temporal features excluded from the
      temporal matrix.

    Proper setup of these attributes is critical for leveraging temporal patterns effectively.

    [See More In Temporal Dependency Guide :fontawesome-solid-timeline:](../../tutorials/temporal_dependency.md){ .md-button }

## ::: scikit_longitudinal.data_preparation.merwav_time_minus.MerWavTimeMinus
    options:
        heading: "MerWavTimeMinus"
        members:
            - get_params
            - _prepare_data
