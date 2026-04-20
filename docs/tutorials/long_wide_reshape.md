---
icon: lucide/git-compare-arrows
---

# Reshaping Longitudinal Data: Long ⇄ Wide with `LongitudinalDataset`

!!! tip "Read this first"
    The [Temporal Dependency tutorial](temporal_dependency.md) explains `features_group` and `non_longitudinal_features`, which are central to how `Sklong` represents longitudinal data. Reading it first will make this tutorial much easier to follow.

`Sklong` works on **wide-format** matrices: one row per subject, one column per (feature, wave) pair. Real-world tabular sources are often stored in **long format**: one row per (subject, wave). The [`LongitudinalDataset`](../API/data_preparation/longitudinal_dataset.md) class ships two methods, `to_wide` and `to_long`, that move between the two layouts. Both validate the inputs, keep `features_group` and `non_longitudinal_features` in sync, and optionally write a CSV alongside.

## The toy cohort

Three patients, three follow-up waves, two longitudinal measurements (`bp`, `chol`), one static covariate (`sex`). The same cohort is used in both directions of the tutorial.

```python
import numpy as np
import pandas as pd

long_df = pd.DataFrame({
    "patient_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
    "wave":       [0, 1, 2, 0, 1, 2, 0, 1, 2],
    "bp":         [120., 122., 121., 130., 131., 128., 110., 112., 115.],
    "chol":       [5.0, 5.5, 6.0, 4.5, np.nan, 5.0, 6.0, 6.0, 6.5],
    "sex":        ["M","M","M","F","F","F","F","F","F"],
})
```

| patient_id | wave | bp  | chol | sex |
|------------|------|-----|------|-----|
| 1 | 0 | 120 | 5.0 | M |
| 1 | 1 | 122 | 5.5 | M |
| 1 | 2 | 121 | 6.0 | M |
| 2 | 0 | 130 | 4.5 | F |
| 2 | 1 | 131 | NaN | F |
| 2 | 2 | 128 | 5.0 | F |
| ... | ... | ... | ... | ... |

## Part 1: Long to Wide

The animation below walks through every long-format row and shows where each value lands in the wide matrix.

<figure class="expandable-media" markdown="span" style="text-align: center;">
 [![Long to wide reshape, animated](../assets/images/tutorials/long_wide_reshape/LongToWide.avif#only-light){ width="100%" loading="lazy" }](../assets/images/tutorials/long_wide_reshape/LongToWide.avif){ .expandable-media__trigger }
 [![Long to wide reshape, animated](../assets/images/tutorials/long_wide_reshape/LongToWideDark.avif#only-dark){ width="100%" loading="lazy" }](../assets/images/tutorials/long_wide_reshape/LongToWideDark.avif){ .expandable-media__trigger }
 <figcaption>Click the image to expand it.</figcaption>
</figure>

The corresponding code is a single call:

```python
from scikit_longitudinal.data_preparation import LongitudinalDataset

dataset = LongitudinalDataset(file_path=None, data_frame=long_df)

wide = dataset.to_wide(
    id_col="patient_id",
    time_col="wave",
    longitudinal_columns=["bp", "chol"],
    static_columns=["sex"],
    wave_format="{feature}_w{wave}",   # default; shown for clarity
    output_path=None,                  # set to a path to dump a CSV
)

print(wide)
print("features_group :", dataset.feature_groups())
print("static columns :", dataset.non_longitudinal_features())
```

Output:

```
            sex  bp_w0  bp_w1  bp_w2  chol_w0  chol_w1  chol_w2
patient_id
1            M  120.0  122.0  121.0      5.0      5.5      6.0
2            F  130.0  131.0  128.0      4.5      NaN      5.0
3            F  110.0  112.0  115.0      6.0      6.0      6.5

features_group : [[1, 2, 3], [4, 5, 6]]
static columns : [0]
```

A few things to notice:

- `wide` has one row per patient, with `static_columns` placed first, then each longitudinal feature expanded across all observed waves (oldest to newest).
- Missing schedule slots are filled with `NaN`. No imputation is performed.
- The dataset's `feature_groups()` and `non_longitudinal_features()` are updated in place, so the wide frame can be passed straight to a `Sklong` estimator or pipeline.

### Common pitfalls

| Situation | What `to_wide` does |
|-----------|---------------------|
| Two rows for the same `(patient_id, wave)` | `ValueError("Duplicate (id, time) rows in long dataframe; ...")` |
| `sex` differs between waves of the same patient | `ValueError("Static columns vary within a subject: ...")` |
| Column listed twice (or as both id/time and longitudinal) | `ValueError("listed more than once")` / `"cannot be a value/static column..."` |
| Empty `longitudinal_columns=[]` | `ValueError("longitudinal_columns must list at least one longitudinal column.")` |

## Part 2: Wide to Long

The other direction, displays as follows:

<figure class="expandable-media" markdown="span" style="text-align: center;">
 [![Wide to long reshape, animated](../assets/images/tutorials/long_wide_reshape/WideToLong.avif#only-light){ width="100%" loading="lazy" }](../assets/images/tutorials/long_wide_reshape/WideToLong.avif){ .expandable-media__trigger }
 [![Wide to long reshape, animated](../assets/images/tutorials/long_wide_reshape/WideToLongDark.avif#only-dark){ width="100%" loading="lazy" }](../assets/images/tutorials/long_wide_reshape/WideToLongDark.avif){ .expandable-media__trigger }
 <figcaption>Click the image to expand it.</figcaption>
</figure>

`to_long` reads the dataset's own `features_group` to drive the un-pivot, so there is no need to re-state the column names. Starting from the same wide cohort built in Part 1 — the `LongitudinalDataset` already carries `features_group` and `non_longitudinal_features` after the `to_wide` call:

```python
from scikit_longitudinal.data_preparation import LongitudinalDataset

dataset = LongitudinalDataset(file_path=None, data_frame=long_df)

wide = dataset.to_wide(
    id_col="patient_id",
    time_col="wave",
    longitudinal_columns=["bp", "chol"],
    static_columns=["sex"],
)

print("features_group :", dataset.feature_groups())
print("static columns :", dataset.non_longitudinal_features())
# features_group : [[1, 2, 3], [4, 5, 6]]
# static columns : [0]

long_again = dataset.to_long(
    feature_base_names=["bp", "chol"],   # column names in the long output
    id_col="patient_id",
    time_col="wave",
    keep_static=True,
    output_path=None,
)
print(long_again.head(8))
```

Output:

```
   patient_id  wave     bp  chol sex
0           1     1  120.0   5.0   M
1           1     2  122.0   5.5   M
2           1     3  121.0   6.0   M
3           2     1  130.0   4.5   F
4           2     2  131.0   NaN   F
5           2     3  128.0   5.0   F
6           3     1  110.0   6.0   F
7           3     2  112.0   6.0   F
```

Notes:

- Wave labels in the long output are positional (1, 2, 3, ...), one slot per group entry.
- `dataset.feature_groups()` and `dataset.non_longitudinal_features()` are reset to `None` because they describe a wide layout that no longer exists.
- `keep_static=False` would drop `sex` from the long frame.

!!! tip "Where next?"
    Review the [`LongitudinalDataset` API reference](../API/data_preparation/longitudinal_dataset.md) for all available parameters, or move on to the [Algorithm Adaptation tutorial](sklong_explore_your_first_estimator.md) to plug your fresh wide dataset into a longitudinal estimator.
