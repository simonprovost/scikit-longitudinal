# Correlation Based Feature Selection Per Group (CFS Per Group)

!!! tip "What is the CFS Per Group module?"
    The `CorrelationBasedFeatureSelectionPerGroup` module implements the CFS-Per-Group algorithm, a longitudinal 
    variant of the standard CFS method. It is designed for feature selection in longitudinal datasets by considering 
    temporal variations across multiple waves (time points). 

    The algorithm operates in two phases: selecting features within each longitudinal group and then refining the 
    selection across all groups and non-longitudinal features.

    We highly recommend reviewing the `Temporal Dependency` page in the documentation for a deeper understanding 
    of feature groups and the `CFS Per Group` module's usage before exploring its API.

    [See The Temporal Dependency Guide :fontawesome-solid-timeline:](../../../tutorials/temporal_dependency.md){ .md-button }

!!! warning "Documentation Under Alpha Construction"
    **This documentation is in its early stages and still being developed.** The API may therefore change, and some parts might be incomplete or inaccurate.

    **Use at your own risk**, and please report anything that seems `incorrect` / `outdated` you find.

    [Open An Issue! :fontawesome-brands-square-github:](https://github.com/simonprovost/scikit-longitudinal/issues){ .md-button }

## ::: scikit_longitudinal.preprocessors.feature_selection.correlation_feature_selection.cfs_per_group.CorrelationBasedFeatureSelectionPerGroup
    options:
        heading: "CorrelationBasedFeatureSelectionPerGroup"
        members:
            - _fit
            - _transform
            - apply_selected_features_and_rename