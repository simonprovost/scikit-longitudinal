# Lexicographical Decision Tree Classifier

!!! tip "What is the Lexicographical Decision Tree Classifier?"
    The Lexicographical Decision Tree Classifier is a specialized machine learning model designed for 
    analyzing longitudinal data, where measurements are collected over time. 

    Unlike traditional decision trees that select splits based solely on statistical measures like information gain, 
    this classifier incorporates the temporal aspect of the data. It prioritizes more recent measurements when deciding
    how to split the data, under the assumption that recent information is often more predictive of outcomes. 

    This is achieved through a lexicographic optimization approach that balances statistical purity with temporal 
    relevance. See further details below.

    We highly recommend reviewing the `Temporal Dependency` page in the documentation for a deeper understanding of 
    feature groups and the `Lexicographical Decision Tree` classifier's usage before exploring its API.

    [See The Temporal Dependency Guide :fontawesome-solid-timeline:](../../../tutorials/temporal_dependency.md){ .md-button }

!!! warning "Documentation Under Alpha Construction"
    **This documentation is in its early stages and still being developed.** The API may therefore change, and some parts
    might be incomplete or inaccurate.

    **Use at your own risk**, and please report anything that seems `incorrect` / `outdated` you find.

    [Open An Issue! :fontawesome-brands-square-github:](https://github.com/simonprovost/scikit-longitudinal/issues){ .md-button }

## ::: scikit_longitudinal.estimators.trees.lexicographical.lexico_decision_tree.LexicoDecisionTreeClassifier
    options:
        heading: "LexicoDecisionTreeClassifier"
        members:
            - fit

!!! note "Where are predict? and predict_proba?"
    The `predict` and `predict_proba` methods are inherited from the `DecisionTreeClassifier` class, which is a part of the `scikit-learn` library.
    Therefore, the `LexicoDecisionTreeClassifier` does not explicitly define these methods. Instead, it inherits them from its parent class.

    Feel free to call them directly on the `LexicoDecisionTreeClassifier` instance.