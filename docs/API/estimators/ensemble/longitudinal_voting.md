# Longitudinal Voting Classifier

!!! tip "What is the Longitudinal Voting Classifier?"
    The Longitudinal Voting Classifier is a versatile ensemble method designed to handle the unique challenges posed
    by longitudinal data. It leverages different voting strategies to combine predictions from multiple base estimators,
    enhancing predictive performance. The base estimators are individually trained, and their predictions are 
    aggregated based on the chosen voting strategy to generate the final prediction.

    Mainly used within SepWav. Relate to this primitive.

!!! warning "Documentation Under Alpha Construction"
    **This documentation is in its early stages and still being developed.** The API may therefore change, and some parts might be incomplete or inaccurate.

    **Use at your own risk**, and please report anything that seems `incorrect` / `outdated` you find.

    [Open An Issue! :fontawesome-brands-square-github:](https://github.com/simonprovost/scikit-longitudinal/issues){ .md-button }

## ::: scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting.LongitudinalVotingClassifier
    options:
        heading: "LongitudinalVotingClassifier"
        members:
            - _fit
            - _predict
            - _predict_proba


!!! note "Use of underscore in method names"
    `_predict` should  be called via `predict` we handle the call to `_predict` in the `predict` method.
    The same applies to `_predict_proba` and `predict_proba`.

## ::: scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting.LongitudinalEnsemblingStrategy
