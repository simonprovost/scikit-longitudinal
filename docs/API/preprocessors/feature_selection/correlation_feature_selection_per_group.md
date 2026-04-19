# Correlation Based Feature Selection Per Group (CFS Per Group)

??? tip "Abstract of CorrelationBasedFeatureSelectionPerGroup"
    *Extracted from Pomsuwan & Freitas (2017), "Feature selection for the classification of longitudinal human ageing data".*

    We propose a new variant of the Correlation-based Feature Selection (CFS) method for coping with longitudinal data - where variables are repeatedly measured across different time points. The proposed CFS variant is evaluated on ten datasets created using data from the English Longitudinal Study of Ageing (ELSA), with different age-related diseases used as the class variables to be predicted. The results show that, overall, the proposed CFS variant leads to better predictive performance than the standard CFS and the baseline approach of no feature selection, when using Naïve Bayes and J48 decision tree induction as classification algorithms (although the difference in performance is very small in the results for J4.8). We also report the most relevant features selected by J48 across the datasets.

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

## ::: scikit_longitudinal.preprocessors.feature_selection.correlation_feature_selection.cfs_per_group.CorrelationBasedFeatureSelectionPerGroup
    options:
        heading: "CorrelationBasedFeatureSelectionPerGroup"
        inherited_members: true
        members:
            - fit
            - transform
            - apply_selected_features_and_rename
