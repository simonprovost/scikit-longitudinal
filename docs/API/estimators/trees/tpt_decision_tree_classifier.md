# Time-penalised Trees Decision Tree Classifier

??? tip "What is the Time-penalised Trees (TpT) Decision Tree Classifier?"
    The `TpTDecisionTreeClassifier` is a longitudinal-aware decision tree that extends the standard CART algorithm with
    a **time-penalised split gain**. At a parent node observed at time $t_p$, a candidate split evaluated at time
    $t_c$ has its information gain $\Delta I$ scaled by an exponential penalty $e^{-\gamma\,(t_c - t_p)}$. The
    splitter therefore prefers earlier waves unless later observations bring a substantially stronger signal, which
    yields sparse-in-time and interpretable trees.

    TpT can consume both `wide` longitudinal matrices (with `features_group`) and LONG-format dataframes (one row per
    `(subject, time)` observation) by setting `assume_long_format=True` and providing `id_col`, `time_col`, and
    `duration_col`.

    We highly recommend reading the `Temporal Dependency` page before exploring the TpT API.

    [See The Temporal Dependency Guide ](../../../tutorials/temporal_dependency.md){ .md-button }

## ::: scikit_longitudinal.estimators.trees.TpT.TpT_decision_tree.TpTDecisionTreeClassifier
    options:
        heading: "TpTDecisionTreeClassifier"
        members:
            - fit
            - predict
            - predict_proba

!!! note "Where do `predict` and `predict_proba` come from?"
    Both methods are inherited from scikit-learn's `DecisionTreeClassifier`. `TpTDecisionTreeClassifier` only overrides
    them to handle the optional LONG→wide conversion; otherwise the standard scikit-learn behaviour applies.

!!! warning "`gamma` vs. `threshold_gain`"
    `threshold_gain` is kept as a backward-compatible alias for `gamma` (both control the time-penalty rate
    $\gamma$). Prefer the explicit `gamma` keyword in new code; if both are provided, `gamma` takes precedence.
