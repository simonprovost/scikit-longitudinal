# Separate Waves Classifier for Longitudinal Data

!!! tip "What is the SepWav module?"
    The `SepWav` module implements the Separate Waves strategy for longitudinal data analysis. 
    It trains individual classifiers on each wave (time point) and combines their predictions using ensemble methods 
    like voting or stacking. This approach leverages temporal information for improved model performance.

    We highly recommend reviewing the `Temporal Dependency` page in the documentation for a deeper understanding of 
    feature groups and the `SepWav` module's usage before exploring its API.

    [See The Temporal Dependency Guide :fontawesome-solid-timeline:](../../tutorials/temporal_dependency.md){ .md-button }

!!! warning "Documentation Under Alpha Construction"
    **This documentation is in its early stages and still being developed.** The API may therefore change, and some parts might be incomplete or inaccurate.

    **Use at your own risk**, and please report anything that seems `incorrect` / `outdated` you find.

    [Open An Issue! :fontawesome-brands-square-github:](https://github.com/simonprovost/scikit-longitudinal/issues){ .md-button }

## ::: scikit_longitudinal.data_preparation.separate_waves.SepWav
    options:
        heading: "SepWav"
        members:
            - get_params
            - fit
            - predict
            - predict_proba
            - predict_wave