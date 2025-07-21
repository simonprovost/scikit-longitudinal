# Merging Waves and Keeping Time Indices for Longitudinal Data

!!! tip "What is the MerWavTimePlus module?"
    The MerWavTimePlus module transforms longitudinal data by merging all features across waves into a single set while
    preserving their time indices. This maintains the temporal structure, enabling longitudinal machine learning methods to
    leverage temporal dependencies and patterns. It provides methods for data preparation and transformation, including
    prepare_data and transform.

    We highly recommend reviewing the `Temporal Dependency` page in the documentation for a deeper understanding of 
    feature groups and the `MerWavTimePlus` module's usage before exploring its API.

    [See The Temporal Dependency Guide :fontawesome-solid-timeline:](../../tutorials/temporal_dependency.md){ .md-button }

!!! warning "Documentation Under Alpha Construction"
    **This documentation is in its early stages and still being developed.** The API may therefore change, and some parts
    might be incomplete or inaccurate.

    **Use at your own risk**, and please report anything that seems `incorrect` / `outdated` you find.

    [Open An Issue! :fontawesome-brands-square-github:](https://github.com/simonprovost/scikit-longitudinal/issues){ .md-button }

## ::: scikit_longitudinal.data_preparation.merwav_time_plus.MerWavTimePlus
    options:
        heading: "MerWavTimePlus"
        members:
            - get_params
            - _prepare_data