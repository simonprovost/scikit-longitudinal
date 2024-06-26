# :book: API Reference :book:

Welcome to the full API documentation of the `Scikit-Longitudinal` toolbox. :toolbox:

## :simple-databricks: Data Preparation
- [Longitudinal Dataset](data_preparation/longitudinal_dataset.md)
- [Aggregation Function (AggrFunc)](data_preparation/aggregation_function.md)
- [Separate Waves (SepWav)](data_preparation/sepwav.md)
- [Merge Waves and discard features 'MervWavTime(-)'](data_preparation/merwav_time_minus.md)
- [Merge Waves and keep features' Time indices 'MervWavTime(+)'](data_preparation/merwav_time_plus.md)

## :simple-jfrogpipelines:Pipeline
- [Longitudinal Pipeline](pipeline.md)

## :wrench: Preprocessors
### :material-selection-search: Feature Selection
- [Correlation Feature Selection Per Group](preprocessors/feature_selection/correlation_feature_selection_per_group.md) 

## :bar_chart: Estimators

### :evergreen_tree: Trees

- [Lexicographical Decision Tree Classifier](estimators/trees/lexico_decision_tree_classifier.md)
- [Lexicographical Decision Tree Regressor](estimators/trees/lexico_decision_tree_regressor.md)

### :material-relation-only-one-to-one-or-many: Ensemble
- [Lexicographical Deep Forest](estimators/ensemble/lexico_deep_forest.md) 
- [Lexicographical Gradient Boosting](estimators/ensemble/lexico_gradient_boosting.md)
- [Lexicographical Random Forest](estimators/ensemble/lexico_random_forest.md)
- [Nested Trees](estimators/ensemble/nested_trees.md) 
- [Longitudinal Stacking](estimators/ensemble/longitudinal_stacking.md) 
- [Longitudinal Voting](estimators/ensemble/longitudinal_voting.md) 

