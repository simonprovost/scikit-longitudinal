# Lexico Random Forest Classifier

!!! tip "What is the Lexico Random Forest Classifier?"
    The Lexico Random Forest Classifier is an advanced ensemble learning model designed specifically for 
    longitudinal data analysis. It extends the traditional Random Forest algorithm by incorporating a 
    lexicographic optimization approach within each decision tree. This approach prioritizes more recent data points 
    (waves) when selecting splits, based on the premise that recent measurements are more predictive and relevant. 

    The classifier is optimized for efficiency using a Cython implementation and is particularly suited for applications
    where temporal recency is critical, such as medical studies or time-series of time-series classification.

    We highly recommend reviewing the `Temporal Dependency` page in the documentation for a deeper understanding of 
    feature groups and the `LexicoRandomForestClassifier`'s usage before exploring its API.

    [See The Temporal Dependency Guide :fontawesome-solid-timeline:](../../../tutorials/temporal_dependency.md){ .md-button }

!!! warning "Documentation Under Alpha Construction"
    **This documentation is in its early stages and still being developed.** The API may therefore change, and some parts might be incomplete or inaccurate.

    **Use at your own risk**, and please report anything that seems `incorrect` / `outdated` you find.

    [Open An Issue! :fontawesome-brands-square-github:](https://github.com/simonprovost/scikit-longitudinal/issues){ .md-button }

## ::: scikit_longitudinal.estimators.ensemble.lexicographical.lexico_random_forest.LexicoRandomForestClassifier
    options:
        heading: "LexicoRandomForestClassifier"
        members:
            - fit

!!! note "Where are predict? and predict_proba?"
    The `predict` and `predict_proba` methods are inherited from the `RandomForestClassifier` class in `scikit-learn`. 
    They are not explicitly defined in `LexicoRandomForestClassifier` but can be called directly on the instance.