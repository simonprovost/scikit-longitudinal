# Aggregation Function for Longitudinal Data

!!! tip "What is the AggrFunc module?"
    The `AggrFunc` module facilitates the application of aggregation functions to feature groups within a longitudinal
    dataset, enabling the use of temporal information before applying traditional machine learning algorithms.

    We highly recommend reviewing the `Temporal Dependency` page in the documentation for a deeper understanding of
    feature groups and the `AggrFunc` module's usage before exploring its API.

    [See The Temporal Dependency Guide :fontawesome-solid-timeline:](../../tutorials/temporal_dependency.md){ .md-button }

!!! warning "Documentation Under Alpha Construction"
    **This documentation is in its early stages and still being developed.** The API may therefore change,
    and some parts might be incomplete or inaccurate.

    **Use at your own risk**, and please report anything that seems `incorrect` / `outdated` you find.

    [Open An Issue! :fontawesome-brands-square-github:](https://github.com/simonprovost/scikit-longitudinal/issues){ .md-button }

## ::: scikit_longitudinal.data_preparation.aggregation_function.AggrFunc
    options:
        heading: "AggrFunc"
        members:
            - get_params
            - _prepare_data
            - _transform