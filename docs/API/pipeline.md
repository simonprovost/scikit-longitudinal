# Longitudinal Pipeline

!!! tip "What is the LongitudinalPipeline module?"
    The `LongitudinalPipeline` module extends scikit-learn's `Pipeline` to handle longitudinal data, ensuring that
    the structure of longitudinal features is updated and maintained throughout transformations. 
    It is designed for longitudinal classification tasks, integrating seamlessly with scikit-learn's ecosystem.

    Let's stack your steps and build a nice LongitudinalPipeline!

    We highly recommend reviewing the `Temporal Dependency` page in the documentation for a deeper understanding of feature groups and the `LongitudinalPipeline` module's usage before exploring its API.

    [See The Temporal Dependency Guide :fontawesome-solid-timeline:](../tutorials/temporal_dependency.md){ .md-button }

!!! warning "Documentation Under Alpha Construction"
    **This documentation is in its early stages and still being developed.** The API may therefore change, and some parts might be incomplete or inaccurate.

    **Use at your own risk**, and please report anything that seems `incorrect` / `outdated` you find.

    [Open An Issue! :fontawesome-brands-square-github:](https://github.com/simonprovost/scikit-longitudinal/issues){ .md-button }

## ::: scikit_longitudinal.pipeline.LongitudinalPipeline
    options:
        heading: "LongitudinalPipeline"
        members:
            - fit
            - predict
            - predict_proba
            - transform
            - fit_transform
