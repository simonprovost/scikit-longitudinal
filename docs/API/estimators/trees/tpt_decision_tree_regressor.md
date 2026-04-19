# Time-penalised Trees Decision Tree Regressor

??? tip "Abstract of TpTDecisionTreeRegressor"
    *Extracted from Valla (2024), "Time-penalised trees (TpT): introducing a new tree-based data mining algorithm for time-varying covariates".*

    This article introduces a new decision tree algorithm that accounts for time-varying covariates in the decision-making process. Traditional decision tree algorithms assume that the covariates are static and do not change over time, which can lead to inaccurate predictions in dynamic environments. Other existing methods suggest workaround solutions such as the pseudo-subject approach. The proposed algorithm utilises a different structure and a time-penalised splitting criterion that allows a recursive partitioning of both the covariates space and time. Relevant historical trends are then inherently involved in the construction of a tree, and are visible and interpretable once it is fit. This approach allows for innovative and highly interpretable analysis in settings where the covariates are subject to change over time. The effectiveness of the algorithm is demonstrated through a real-world data application in life insurance. The results presented in this article can be seen as an introduction or proof-of-concept of the time-penalised approach, and the algorithm's theoretical properties and comparison against existing approaches on datasets from various fields will be explored in forthcoming work.

    Adapted to regression, this estimator applies the same time-penalised splitting criterion above inside `DecisionTreeRegressor`, replacing the classification impurity improvement with variance reduction (MSE) before applying the exponential time penalty.

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

## ::: scikit_longitudinal.estimators.trees.TpT.TpT_decision_tree_regressor.TpTDecisionTreeRegressor
    options:
        heading: "TpTDecisionTreeRegressor"
        inherited_members: true
        members:
            - fit
            - predict
