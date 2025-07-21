# Nested Trees Classifier

!!! tip "What is the Nested Trees Classifier?"
    The Nested Trees Classifier is a unique and innovative classification algorithm specifically designed for 
    longitudinal datasets. It enhances traditional decision tree algorithms by embedding smaller decision trees within
    the nodes of a primary tree structure, leveraging the inherent information in longitudinal data optimally.

    We highly recommend reviewing the `Temporal Dependency` page in the documentation for a deeper understanding of 
    feature groups and the `NestedTreesClassifier`'s usage before exploring its API.

    [See The Temporal Dependency Guide :fontawesome-solid-timeline:](../../../tutorials/temporal_dependency.md){ .md-button }

!!! warning "Documentation Under Alpha Construction"
    **This documentation is in its early stages and still being developed.** The API may therefore change, and some parts might be incomplete or inaccurate.

    **Use at your own risk**, and please report anything that seems `incorrect` / `outdated` you find.

    [Open An Issue! :fontawesome-brands-square-github:](https://github.com/simonprovost/scikit-longitudinal/issues){ .md-button }

## ::: scikit_longitudinal.estimators.ensemble.nested_trees.nested_trees.NestedTreesClassifier
    options:
        heading: "NestedTreesClassifier"
        members:
            - _fit
            - _predict
            - _predict_proba
            - print_nested_tree

!!! note "Inherited Methods"
    The `predict` and `predict_proba` methods are inherited from the `CustomClassifierMixinEstimator` 
    class and can be called directly on the `NestedTreesClassifier` instance. 

    They internally call `_predict` and `_predict_proba` respectively.