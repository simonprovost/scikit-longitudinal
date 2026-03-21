---
icon: lucide/binary
---

# Binary vs. Multiclass Classification

!!! tip "Dataset Used in Tutorials"
    Use the shared synthetic dataset defined in the [tutorials overview](overview.md#dataset-used-in-tutorials). Generate it once there and reuse it here.

The same `Sklong` workflow supports both binary and multiclass classification. In practice, the main changes are:

| Aspect | Binary classification | Multiclass classification |
|---|---|---|
| Target labels | Two labels such as `0` and `1` | Three or more labels such as `0`, `1`, and `2` |
| `predict_proba` shape | `(n_samples, 2)` | `(n_samples, n_classes)` |
| `classes_` | Two class labels | One entry per class |
| AUPRC | Usually computed from the positive-class scores | Usually computed with `macro`, `weighted`, or `micro` averaging |

The estimators below support both binary and multiclass targets:

- `LexicoDecisionTreeClassifier`
- `LexicoRandomForestClassifier`
- `LexicoGradientBoostingClassifier`
- `LexicoDeepForestClassifier`
- `NestedTreesClassifier`
- `SepWav` with voting or stacking

## Step 1: Binary classification

This first example uses the original `stroke_w2` target from the tutorial dataset.

```python
from sklearn.metrics import accuracy_score

from scikit_longitudinal import auprc_score
from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.estimators.ensemble import LexicoRandomForestClassifier

binary_dataset = LongitudinalDataset("./extended_stroke_longitudinal.csv")
binary_dataset.load_data_target_train_test_split(
 target_column="stroke_w2",
 test_size=0.2,
 random_state=42,
)
binary_dataset.setup_features_group([[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]])

binary_clf = LexicoRandomForestClassifier(
 features_group=binary_dataset.feature_groups(),
 n_estimators=100,
 random_state=42,
)
binary_clf.fit(binary_dataset.X_train, binary_dataset.y_train)

binary_pred = binary_clf.predict(binary_dataset.X_test)
binary_proba = binary_clf.predict_proba(binary_dataset.X_test)

print(binary_clf.classes_) # Example: [0 1]
print(binary_proba.shape) # Example: (100, 2)
print(accuracy_score(binary_dataset.y_test, binary_pred))
print(auprc_score(binary_dataset.y_test, binary_proba[:, 1]))
```

!!! note
    In the binary case, `predict_proba` returns two columns ordered according to `classes_`. When you compute AUPRC from a one-dimensional score vector, pass the scores for the positive class, which is usually the second column.

## Step 2: Create a multiclass target from the same longitudinal table

To compare like for like, we can derive a three-class risk target from the same wave-2 measurements.

```python
import pandas as pd

df = pd.read_csv("./extended_stroke_longitudinal.csv")

risk_score = (
 df["smoke_w2"]
 + df["cholesterol_w2"]
 + df["blood_pressure_w2"]
 + df["diabetes_w2"]
)

df["risk_group_w2"] = pd.cut(
 risk_score,
 bins=[-1, 0, 2, 4],
 labels=[0, 1, 2],
).astype(int)

df.to_csv("./extended_stroke_multiclass_longitudinal.csv", index=False)
```

Here the derived classes are:

- `0`: low risk
- `1`: medium risk
- `2`: high risk

## Step 3: Multiclass classification

The fitting workflow stays almost identical. The main difference is that the target now contains three labels and `predict_proba` returns three columns.

```python
from sklearn.metrics import accuracy_score, classification_report

from scikit_longitudinal import auprc_score
from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.estimators.ensemble import LexicoRandomForestClassifier

multiclass_dataset = LongitudinalDataset("./extended_stroke_multiclass_longitudinal.csv")
multiclass_dataset.load_data_target_train_test_split(
 target_column="risk_group_w2",
 test_size=0.2,
 random_state=42,
)
multiclass_dataset.setup_features_group([[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]])

multiclass_clf = LexicoRandomForestClassifier(
 features_group=multiclass_dataset.feature_groups(),
 n_estimators=100,
 random_state=42,
)
multiclass_clf.fit(multiclass_dataset.X_train, multiclass_dataset.y_train)

multiclass_pred = multiclass_clf.predict(multiclass_dataset.X_test)
multiclass_proba = multiclass_clf.predict_proba(multiclass_dataset.X_test)

print(multiclass_clf.classes_) # Example: [0 1 2]
print(multiclass_proba.shape) # Example: (100, 3)
print(accuracy_score(multiclass_dataset.y_test, multiclass_pred))
print(classification_report(multiclass_dataset.y_test, multiclass_pred))
print(auprc_score(multiclass_dataset.y_test, multiclass_proba, average="macro"))
```

!!! note
    In the multiclass case, `auprc_score` expects the full two-dimensional score matrix and an averaging strategy such as `macro`, `weighted`, `micro`, or `None`.

## Step 4: The same multiclass target also works with `SepWav`

If you prefer wave-wise ensembling, the multiclass target can also be used with `SepWav`.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from scikit_longitudinal.data_preparation import SepWav
from scikit_longitudinal.estimators.ensemble.longitudinal_voting.longitudinal_voting import (
 LongitudinalEnsemblingStrategy,
)

sepwav = SepWav(
 estimator=RandomForestClassifier(max_depth=5, random_state=42),
 features_group=multiclass_dataset.feature_groups(),
 non_longitudinal_features=multiclass_dataset.non_longitudinal_features(),
 feature_list_names=multiclass_dataset.data.columns.tolist(),
 voting=LongitudinalEnsemblingStrategy.STACKING,
 stacking_meta_learner=LogisticRegression(max_iter=200),
)

sepwav.fit(multiclass_dataset.X_train, multiclass_dataset.y_train)
sepwav_proba = sepwav.predict_proba(multiclass_dataset.X_test)

print(sepwav.classes_) # Example: [0 1 2]
print(sepwav_proba.shape) # Example: (100, 3)
```
