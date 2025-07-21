# Lexicographical Decision Tree Regressor

!!! tip "What is the Lexicographical Decision Tree Regressor?"
    The Lexicographical Decision Tree Regressor is a specialized machine learning model designed for 
    analyzing longitudinal data, where measurements are collected over time. 

    Unlike traditional decision trees regressor that select splits based solely on statistical measures like MSE, 
    this Regressor incorporates the temporal aspect of the data. It prioritizes more recent measurements when deciding
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

## ::: scikit_longitudinal.estimators.trees.lexicographical.lexico_decision_tree_regressor.LexicoDecisionTreeRegressor
    options:
        heading: "LexicoDecisionTreeRegressor"
        members:
            - fit

!!! note "Where is predict?"
    The `predict` method is inherited from the `DecisionTreeRegressor` class, which is a part of the `scikit-learn` library.
    Therefore, the `LexicoDecisionTreeRegressor` does not explicitly define this method. Instead, it inherits them from its parent class.

    Feel free to call it directly on the `LexicoDecisionTreeRegressor` instance.