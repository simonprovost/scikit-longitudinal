# :book: API Reference :book:

# API reference :toolbox:

## :book: Overview of each module

<div class="grid cards" markdown>

-   :material-cable-data:{ .lg .middle } __Data Preparation__

    ---

    `Longitudinal tabular data` are looking similarly than other `non-Longitudinal tabular data` at first, but
    there are temporal information throughout either the attributes (_wide_) or repetitive rows (_long_). 
    
    The data preparation primitives are designed to help you load your tabular longitudinal data, **the right way!**

    [:octicons-arrow-right-24: See the primitives](#data-preparation)

-   :material-data-matrix-scan:{ .lg .middle } __Data Transformation__

    ---

    The `data transformation` primitives are designed to start transforming your `longitudinal data` to either 
    be ready for (A) standard machine learning primitives or (B) longitudinal-based machine learning ones.

    [:octicons-arrow-right-24: See the primitives](#data-transformation)

-   :material-selection-search:{ .lg .middle } __Preprocessors__

    ---

    The `preprocessors` are designed to help you select the right features for your `longitudinal data`. 
    They are primarily longitudinal-based primitives, for standard-based ones, please refer to `sklearn` or other alike
    libraries.

    [:octicons-arrow-right-24: See the primitives](#preprocessors)

-   :bar_chart:{ .lg .middle } __Estimators__

    ---

    The `estimators` are designed to help you train your `longitudinal data`. They are primarily 
    longitudinal-based primitives, for standard-based ones, please refer to `sklearn` or other alike libraries.

    [:octicons-arrow-right-24: See the primitives](#estimators)

</div>

___________________

## :material-cable-data: Data Preparation
- [Longitudinal Dataset](data_preparation/longitudinal_dataset.md)

## :material-data-matrix-scan: Data Transformation
- [Merge Waves and discard features 'MervWavTime(-)'](data_preparation/merwav_time_minus.md)
- [Merge Waves and keep features' Time indices 'MervWavTime(+)'](data_preparation/merwav_time_plus.md)
- [Aggregation Function (AggrFunc)](data_preparation/aggregation_function.md)
- [Separate Waves (SepWav)](data_preparation/sepwav.md)

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

## :simple-jfrogpipelines: Pipeline
- [Longitudinal Pipeline](pipeline.md)

<details>
<summary>Dataset Used in Examples</summary>

The following is a made-up longitudinal dataset (ELSA-style) used in the examples throughout this documentation. 
It is not based on real data and is provided for illustrative purposes only:

```csv
smoke_w1,smoke_w2,cholesterol_w1,cholesterol_w2,age,gender,stroke_w2
0,1,0,1,45,1,0
1,1,1,1,50,0,1
0,0,0,0,55,1,0
1,1,1,1,60,0,1
0,1,0,1,65,1,0
0,1,0,1,45,1,0
1,1,1,1,50,0,1
0,0,0,0,55,1,0
1,1,1,1,60,0,1
0,1,0,1,65,1,0
0,1,0,1,45,1,0
1,1,1,1,50,0,1
0,0,0,0,55,1,0
1,1,1,1,60,0,1
0,1,0,1,65,1,0
0,1,0,1,45,1,0
1,1,1,1,50,0,1
0,0,0,0,55,1,0
1,1,1,1,60,0,1
```
</details>