# Lexico Gradient Boosting Classifier

!!! tip "What is the Lexico Gradient Boosting Classifier?"
    The Lexico Gradient Boosting Classifier is an advanced ensemble learning model tailored for longitudinal data analysis.
    It combines the power of gradient boosting with a lexicographic optimization approach to prioritize more recent 
    data points (waves) in its decision-making process. This makes it particularly effective for datasets where temporal
    recency is crucial, such as medical studies or time-series classification. 

    The classifier uses Lexico Decision Tree Regressors as base estimators, which are specialized to handle 
    the temporal structure of longitudinal data.

!!! warning "Documentation Under Alpha Construction"
    **This documentation is in its early stages and still being developed.** The API may therefore change, and some parts might be incomplete or inaccurate.

    **Use at your own risk**, and please report anything that seems `incorrect` / `outdated` you find.

    [Open An Issue! :fontawesome-brands-square-github:](https://github.com/simonprovost/scikit-longitudinal/issues){ .md-button }

## ::: scikit_longitudinal.estimators.ensemble.lexicographical.lexico_gradient_boosting.LexicoGradientBoostingClassifier
    options:
        heading: "LexicoGradientBoostingClassifier"
        members:
            - _fit
            - _predict
            - _predict_proba
            - feature_importances_

!!! note "Use of underscore in method names"
    `_predict` should  be called via `predict` we handle the call to `_predict` in the `predict` method.
    The same applies to `_predict_proba` and `predict_proba`.