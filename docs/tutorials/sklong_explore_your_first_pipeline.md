---
icon: lucide/workflow
---

# Pipelines: Mix Longitudinal Components

!!! tip "Dataset Used in Tutorials"
    Use the shared synthetic dataset defined in the [tutorials overview](overview.md#dataset-used-in-tutorials). Generate it once there and reuse it here.

Pipelines let you chain longitudinal transformations, preprocessing, and estimation in one interface.

The animation below illustrates the key contract that makes this work: `features_group` travels alongside `X` through every step, and `update_feature_groups_callback` rewrites those indices whenever a step reshapes the matrix — so the final estimator still sees a coherent temporal structure.

<figure class="expandable-media" markdown="span" style="text-align: center;">
 [![LongitudinalPipeline propagates features_group between steps](../assets/images/tutorials/sklong_explore_your_first_pipeline/PipelineChain.avif#only-light){ width="100%" loading="lazy" }](../assets/images/tutorials/sklong_explore_your_first_pipeline/PipelineChain.avif){ .expandable-media__trigger }
 [![LongitudinalPipeline propagates features_group between steps](../assets/images/tutorials/sklong_explore_your_first_pipeline/PipelineChainDark.avif#only-dark){ width="100%" loading="lazy" }](../assets/images/tutorials/sklong_explore_your_first_pipeline/PipelineChainDark.avif){ .expandable-media__trigger }
 <figcaption>Click the image to expand it.</figcaption>
</figure>

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
 ('MerWavTimePlus', MerWavTimePlus()), # Merge waves keeping time indices
 ('CFSPerGroup', CorrelationBasedFeatureSelectionPerGroup()), # Longitudinal feature selection
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
