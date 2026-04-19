# Separate Waves Classifier

??? tip "Abstract of Separate Waves (SepWav)"
    *Extracted from "A New Longitudinal Classification Method Based on Stacking Predictions for Separate Time Points" (BCS SGAI AI-2025).*

    Biomedical research often uses longitudinal data with repeated measurements of variables across time (e.g. cholesterol measured across time), which is challenging for standard machine learning algorithms due to intrinsic temporal dependencies. The Separate Waves (SepWav) data-transformation method trains a base classifier for each time point ("wave") and aggregates their predictions via voting. However, the simplicity of the voting mechanism may not be enough to capture complex patterns of time-dependent interactions involving the base classifiers' predictions. Hence, we propose a novel SepWav method where the simple voting mechanism is replaced by a stacking-based meta-classifier that integrates the base classifiers' wave-specific predictions into a final predicted class label, aiming at improving predictive performance. Experiments with 20 datasets of ageing-related diseases have shown that, overall, the proposed Stacking-based SepWav method achieved significantly better predictive performance than two other methods for longitudinal classification in most cases, when using class-weight adjustment as a class-balancing method.

    [See More In References :fontawesome-solid-book:](../../publications.md){ .md-button }

## ::: scikit_longitudinal.data_preparation.separate_waves.SepWav
    options:
        heading: "SepWav"
        inherited_members: true
        members:
            - get_params
            - fit
            - predict
            - predict_proba
            - predict_wave

---

## SepWav ensemble back-ends

`SepWav` delegates the final aggregation of per-wave predictions to one of the
two classifiers below.

### Longitudinal Voting Classifier

Aggregates per-wave predictions with a configurable voting rule: simple
majority, linear or exponential recency decay, or cross-validation-weighted
voting.

#### ::: scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting.LongitudinalVotingClassifier
    options:
        heading: "LongitudinalVotingClassifier"
        inherited_members: true
        members:
            - fit
            - predict
            - predict_proba

#### ::: scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting.LongitudinalEnsemblingStrategy

### Longitudinal Stacking Classifier

Trains a meta-learner on the class probabilities emitted by the per-wave
classifiers fitted by `SepWav`.

#### ::: scikit_longitudinal.estimators.ensemble.longitudinal_stacking.longitudinal_stacking.LongitudinalStackingClassifier
    options:
        heading: "LongitudinalStackingClassifier"
        inherited_members: true
        members:
            - fit
            - predict
            - predict_proba

