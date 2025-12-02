# 🔗 Pipelines: Mix Longitudinal Components

!!! important "Dataset Used in Tutorials"
    Generate `extended_stroke_longitudinal.csv` once using the snippet in the [tutorials overview](overview.md#dataset-used-in-tutorials), then reuse it here.

!!! tip "Prerequisite Reading"
    Ensure you've read the [Temporal Dependency Guide](temporal_dependency.md) and [Data Format Tutorial](sklong_longitudinal_data_format.md).

Pipelines let you chain longitudinal transformations, preprocessing, and estimation in one interface.

## Load and prepare the dataset

Using `extended_stroke_longitudinal.csv`:

```python
from scikit_longitudinal.data_preparation import LongitudinalDataset

dataset = LongitudinalDataset('./extended_stroke_longitudinal.csv')
dataset.load_data_target_train_test_split(target_column='stroke_w2', test_size=0.2, random_state=42)
dataset.setup_features_group([[2,3], [4,5], [6,7], [8,9], [10,11], [12,13]])
```

## Example 1: Data-preparation pipeline (flatten, then scikit-learn)

Combine a flattening step with familiar scikit-learn transformers and estimators:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from scikit_longitudinal.data_preparation import AggrFunc
from scikit_longitudinal.pipeline import LongitudinalPipeline

steps = [
    ('aggr_mean', AggrFunc(
        features_group=dataset.feature_groups(),
        non_longitudinal_features=dataset.non_longitudinal_features(),
        feature_list_names=dataset.data.columns.tolist(),
        aggregation_func='mean',
    )),
    ('scale', StandardScaler()),
    ('log_reg', LogisticRegression(max_iter=500)),
]

pipeline = LongitudinalPipeline(
    steps=steps,
    features_group=dataset.feature_groups(),
    non_longitudinal_features=dataset.non_longitudinal_features(),
    feature_list_names=dataset.data.columns.tolist(),
    update_feature_groups_callback='default'
)

pipeline.fit(dataset.X_train, dataset.y_train)
y_pred = pipeline.predict(dataset.X_test)
print(f"Accuracy (data-prep path): {accuracy_score(dataset.y_test, y_pred):.3f}")
```

## Example 2: Algorithm-adaptation pipeline (preserve temporal dependency)

Keep temporal structure intact and use longitudinal-aware primitives:

```python
from sklearn.metrics import accuracy_score

from scikit_longitudinal.pipeline import LongitudinalPipeline
from scikit_longitudinal.data_preparation import MerWavTimePlus
from scikit_longitudinal.preprocessors.feature_selection import CorrelationBasedFeatureSelectionPerGroup
from scikit_longitudinal.estimators.trees import LexicoDecisionTreeClassifier

steps = [
    ('MerWavTimePlus', MerWavTimePlus()),  # Merge waves keeping time indices
    ('CFSPerGroup', CorrelationBasedFeatureSelectionPerGroup()),  # Longitudinal feature selection
    ('LexicoDT', LexicoDecisionTreeClassifier(threshold_gain=0.01, random_state=42))
]

pipeline = LongitudinalPipeline(
    steps=steps,
    features_group=dataset.feature_groups(),
    non_longitudinal_features=dataset.non_longitudinal_features(),
    feature_list_names=dataset.data.columns.tolist(),
    update_feature_groups_callback='default'
)

pipeline.fit(dataset.X_train, dataset.y_train)
y_pred = pipeline.predict(dataset.X_test)
print(f"Accuracy (algorithm-adaptation path): {accuracy_score(dataset.y_test, y_pred):.3f}")
```

Pipelines chain steps seamlessly, letting you choose whether to flatten or retain temporal dependencies while keeping a consistent interface.
