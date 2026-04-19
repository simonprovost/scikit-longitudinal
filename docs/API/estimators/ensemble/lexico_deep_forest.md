# Lexico Deep Forest Classifier

??? tip "Abstract of LexicoDeepForestClassifier"
    *Extracted from Ribeiro & Freitas (2024), "Lexicographical random forests for longitudinal data classification".*

    Standard supervised machine learning methods often ignore the temporal information represented in longitudinal data, but that information can lead to more precise predictions in classification tasks. Data preprocessing techniques and classification algorithms can be adapted to cope directly with longitudinal data inputs, making use of temporal information such as the time-index of features and previous measurements of the class variable. In this article, we propose two changes to the classification task of predicting age-related diseases in a real-world dataset created from the English Longitudinal Study of Ageing. First, we explore the addition of previous measurements of the class variable, and estimating the missing data in those added features using intermediate classifiers. Second, we propose a new split-feature selection procedure for a random forest's decision trees, which considers the candidate features' time-indexes, in addition to the information gain ratio. Our experiments compared the proposed approaches to baseline approaches, in 3 prediction scenarios, varying the "time gap" for the prediction - how many years in advance the class (occurrence of an age-related disease) is predicted. The experiments were performed on 10 datasets varying the class variable, and showed that the proposed approaches increased the random forest's predictive accuracy.

    Adapted and integrated into a Deep Forest cascade, this estimator stacks layers of `LexicoRandomForestClassifier`s (and optional diversity learners) so that each layer applies the lexicographic split-selection procedure above while propagating wave-aware predictions through the cascade.

    [See More In References :fontawesome-solid-book:](../../../publications.md){ .md-button }

??? question "What are features_group and non_longitudinal_features?"
    Two key attributes, `features_group` and `non_longitudinal_features`, enable algorithms to interpret the
    temporal structure of longitudinal data.

    - **features_group**: A list of lists where each sublist contains indices of a longitudinal attribute's
      waves, ordered from oldest to most recent. This captures temporal dependencies.
    - **non_longitudinal_features**: A list of indices for static, non-temporal features excluded from the
      temporal matrix.

    Proper setup of these attributes is critical for leveraging temporal patterns effectively.

    [See More In Temporal Dependency Guide :fontawesome-solid-timeline:](../../../tutorials/temporal_dependency.md){ .md-button }

## ::: scikit_longitudinal.estimators.ensemble.lexicographical.lexico_deep_forest.LexicoDeepForestClassifier
    options:
        heading: "LexicoDeepForestClassifier"
        inherited_members: true
        members:
            - fit
            - predict
            - predict_proba

## ::: scikit_longitudinal.estimators.ensemble.lexicographical.lexico_deep_forest.LongitudinalClassifierType

## ::: scikit_longitudinal.estimators.ensemble.lexicographical.lexico_deep_forest.LongitudinalEstimatorConfig
