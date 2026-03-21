# API

Welcome to `Sklong`'s API where you will find all references to each and one of the available modules of Scikit-Longitudinal, hyperparameters, examples and more. Enjoy the wander!

??? warning "Documentation Under Alpha Construction"
    Some API reference pages are still being refined. If something looks incomplete or outdated, please [open an issue](https://github.com/simonprovost/scikit-longitudinal/issues).

<div class="grid cards" markdown>

- __Data Preparation__

    ---

    The data preparation primitives help you load, inspect, and organise your `longitudinal data` before any downstream transformation or modelling step.

    [Jump to primitives](data_preparation/longitudinal_dataset.md)

- __Data Transformation__

    ---

    The `data transformation` primitives are designed to start transforming your `longitudinal data` to either
    be ready for (A) standard machine learning primitives or (B) longitudinal-based machine learning ones.

    [Jump to primitives](data_preparation/merwav_time_minus.md)

- __Preprocessors__

    ---

    Preprocess your `longitudinal data` prior to perform machine learning. As of today, only `Feature Selection` for `longitudinal data` are available.

    [Jump to primitives](preprocessors/feature_selection/correlation_feature_selection_per_group.md)

- __Estimators__

    ---

    `Classifiers` are designed to train on your `longitudinal data` to predict binary or multi-class label(s).

    [Jump to primitives](estimators/trees/lexico_decision_tree_classifier.md)

- __Pipeline__

    ---

    The `pipeline` primitives help you chain preparation, preprocessing, and modelling steps while preserving longitudinal metadata across the workflow.

    [Jump to primitives](pipeline/longitudinal_pipeline.md)

</div>

!!! question "Looking for credits and paper references?"
    Algorithm-specific citations and contributor attributions live in [Publications](../publications.md).
