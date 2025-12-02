# 🧪 Data Preparation: Flatten Temporal Dependency for Scikit-Learn Estimators

!!! important "Dataset Used in Tutorials"
    Generate `extended_stroke_longitudinal.csv` once using the snippet in the [tutorials overview](overview.md#dataset-used-in-tutorials), then reuse it here.

!!! tip "Prerequisite Reading"
    Start with the [Temporal Dependency](temporal_dependency.md) and [Longitudinal Data Format](sklong_longitudinal_data_format.md) guides so you know how to describe waves and non-longitudinal features.

Data-preparation workflows flatten longitudinal structure so you can plug the output into standard `scikit-learn` estimators. Follow this step-by-step path with [`AggrFunc`](../API/data_preparation/aggregation_function.md) (mean aggregation) and `LogisticRegression`—no longitudinal-specific pipeline required.

## Step 1: Load data and define temporal dependencies

```python
from scikit_longitudinal.data_preparation import LongitudinalDataset

dataset = LongitudinalDataset('./extended_stroke_longitudinal.csv')
dataset.load_data_target_train_test_split(target_column='stroke_w2', test_size=0.2, random_state=42)
dataset.setup_features_group([[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]])
```

## Step 2: Flatten with `AggrFunc`

```python
from scikit_longitudinal.data_preparation import AggrFunc

aggregator = AggrFunc(
    features_group=dataset.feature_groups(),
    non_longitudinal_features=dataset.non_longitudinal_features(),
    feature_list_names=dataset.data.columns.tolist(),
    aggregation_func="mean",
)

X_train_flat = aggregator.fit_transform(dataset.X_train)
X_test_flat = aggregator.transform(dataset.X_test)
```

## Step 3: Train and evaluate a scikit-learn estimator

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

clf = LogisticRegression(max_iter=500)
clf.fit(X_train_flat, dataset.y_train)
y_pred = clf.predict(X_test_flat)

print(classification_report(dataset.y_test, y_pred))
```

`AggrFunc` outputs a static tabular matrix, which `LogisticRegression` can train on using the familiar Fit—Predict API. Swap in any other standard estimator (e.g., `RandomForestClassifier`) once the flattening step is in place.

!!! tip "Explore more data-preparation options"
    Find additional flattening strategies and parameters in the [API reference](../API/index.md#data-preparation).
