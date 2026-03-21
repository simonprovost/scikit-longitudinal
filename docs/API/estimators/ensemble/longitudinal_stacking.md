# Longitudinal Stacking Classifier

??? tip "What is the Longitudinal Stacking Classifier?"
    The Longitudinal Stacking Classifier is a sophisticated ensemble method designed to address the complexities of longitudinal data. It employs a stacking approach, combining predictions from multiple pre-trained base estimators to serve as input features for a meta-learner, which generates the final prediction. This classifier is particularly effective for capturing temporal dependencies and enhancing predictive performance in longitudinal datasets, especially when used with the "SepWav" (Separate Waves) strategy.

    We highly recommend reviewing the `Temporal Dependency` page in the documentation for a deeper understanding of feature groups and the `LongitudinalStackingClassifier`'s usage before exploring its API.

    [See The Temporal Dependency Guide ](../../../tutorials/temporal_dependency.md){ .md-button }

## ::: scikit_longitudinal.estimators.ensemble.longitudinal_stacking.longitudinal_stacking.LongitudinalStackingClassifier
    options:
        heading: "LongitudinalStackingClassifier"
        members:
            - _fit
            - _predict
            - _predict_proba

!!! note "Use of underscore in method names"
    `_predict` should be called via `predict`; we handle the call to `_predict` in the `predict` method.
    The same applies to `_predict_proba` and `predict_proba`.
