# 🔍 Algorithm Adaptation: Preserve Temporal Dependency for Sklong Estimators

!!! important "Dataset Used in Tutorials"
    Generate `extended_stroke_longitudinal.csv` once using the snippet in the [tutorials overview](overview.md#dataset-used-in-tutorials), then reuse it here.

!!! tip "Prerequisite Reading"
    Ensure you've read the [Temporal Dependency Guide](temporal_dependency.md) and [Data Format Tutorial](sklong_longitudinal_data_format.md).

Algorithm-adaptation workflows keep temporal structure intact. This walkthrough uses [`LexicoDecisionTreeClassifier`](../API/estimators/trees/lexico_decision_tree_classifier.md), which prioritises recent waves while respecting the full sequence.

## Step 1: Load and prepare data

```python
from scikit_longitudinal.data_preparation import LongitudinalDataset

dataset = LongitudinalDataset('./extended_stroke_longitudinal.csv')
dataset.load_data_target_train_test_split(target_column='stroke_w2', test_size=0.2, random_state=42)
dataset.setup_features_group([[2,3], [4,5], [6,7], [8,9], [10,11], [12,13]])
```

## Step 2: Initialize and fit the estimator

```python
from scikit_longitudinal.estimators.trees import LexicoDecisionTreeClassifier

clf = LexicoDecisionTreeClassifier(
    features_group=dataset.feature_groups(),
    threshold_gain=0.01,
    random_state=42
)

clf.fit(dataset.X_train, dataset.y_train)
```

## Step 3: Predict and evaluate

```python
y_pred = clf.predict(dataset.X_test)
print(y_pred)  # Example output

from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(dataset.y_test, y_pred)}")
```

This introduces basic estimator usage. Experiment with hyperparameters like `threshold_gain`.

!!! tip "Explore more longitudinal-aware estimators"
    Review the estimator catalog and parameters in the [API reference](../API/index.md#estimators) for additional options.
