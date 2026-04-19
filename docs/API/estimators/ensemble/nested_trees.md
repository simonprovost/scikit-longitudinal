# Nested Trees Classifier

??? tip "Abstract of NestedTreesClassifier"
    *Extracted from Ovchinnik, Otero & Freitas (2022), "Nested trees for longitudinal classification".*

    Longitudinal datasets contain repeated measurements of the same variables at different points in time. Longitudinal data mining algorithms aim to utilize such datasets to extract interesting knowledge and produce useful models. Many existing longitudinal classification methods either dismiss the longitudinal aspect of the data during model construction or produce complex models that are scarcely interpretable. We propose a new longitudinal classification algorithm based on decision trees, named Nested Trees. It utilizes a unique longitudinal model construction method that is fully aware of the longitudinal aspect of the predictive attributes (variables) and constructs tree nodes that make decisions based on a longitudinal attribute as a whole, considering measurements of that attribute across multiple time points. The algorithm was evaluated using 10 classification tasks based on the English Longitudinal Study of Ageing (ELSA) data.

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

## ::: scikit_longitudinal.estimators.ensemble.nested_trees.nested_trees.NestedTreesClassifier
    options:
        heading: "NestedTreesClassifier"
        inherited_members: true
        members:
            - fit
            - predict
            - predict_proba
            - print_nested_tree
