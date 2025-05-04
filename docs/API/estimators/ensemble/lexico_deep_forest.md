# Lexico Deep Forest Classifier

!!! tip "What is the Lexico Deep Forest Classifier?"
    The Lexico Deep Forest Classifier is an advanced ensemble learning model designed for longitudinal data analysis.
    It extends the Deep Forest framework by incorporating longitudinal-adapted base estimators that capture temporal 
    complexities and interdependencies inherent in longitudinal data. The classifier combines accurate learners 
    (longitudinal base estimators) and weak learners (diversity non-longitudinal estimators) to improve robustness 
    and generalization, making it ideal for applications like medical studies or time-series classification.

    The classifier uses Lexico Random Forest classifiers as base estimators, which are specialized to handle 
    the temporal structure of longitudinal data.

!!! warning "Documentation Under Alpha Construction"
    **This documentation is in its early stages and still being developed.** The API may therefore change, and some parts might be incomplete or inaccurate.

    **Use at your own risk**, and please report anything that seems `incorrect` / `outdated` you find.

    [Open An Issue! :fontawesome-brands-square-github:](https://github.com/simonprovost/scikit-longitudinal/issues){ .md-button }

## ::: scikit_longitudinal.estimators.ensemble.lexicographical.lexico_deep_forest.LexicoDeepForestClassifier
    options:
        heading: "LexicoDeepForestClassifier"
        members:
            - _fit
            - _predict
            - _predict_proba

!!! note "Use of underscore in method names"
    `_predict` should be called via `predict` we handle the call to `_predict` in the `predict` method.
    The same applies to `_predict_proba` and `predict_proba`.

## ::: scikit_longitudinal.estimators.ensemble.lexicographical.lexico_deep_forest.LongitudinalClassifierType

## ::: scikit_longitudinal.estimators.ensemble.lexicographical.lexico_deep_forest.LongitudinalEstimatorConfig
