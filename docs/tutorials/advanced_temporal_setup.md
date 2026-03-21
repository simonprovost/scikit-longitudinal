---
icon: lucide/compass
---

# Advanced Feature Group (Temporal) Setup for Uneven Waves

Longitudinal datasets often have **uneven observation counts** per subject (e.g., some patients attend 2 visits while
others have 4). In `Sklong`, the recommended approach is:

1. **Represent data in wide format** (one row per subject).
2. **Include columns for the maximum number of waves** you want to model.
3. **Use missing values (`NaN`) for subjects who do not have that wave**.
4. **Define `features_group` with padding (`-1`) only when a whole wave column is missing** for a feature.

This tutorial shows how to do that in practice.

## Step 1: Start from a long table (uneven observations) —— Optional

A long-format dataset makes it easy to describe variable visit counts, but it is not what `Sklong` consumes directly.
We show this first step for clarity, but if you already have your data in wide format, you can skip to Step 3.

!!! note "`Waves` Means Visit Number"
    Note also, that `wave` means visit number here (1, 2, 3, ...). It could be any time interval between visits, even irregular ones.
    Sklong mainly cares about the distance between visits, not the actual time interval. This could change in future versions,
    if new primitives are added to handle specifically time (irregular mostly) intervals.

Below, some subjects are missing wave `2`:

```python
import pandas as pd

long_df = pd.DataFrame(
 {
 "subject_id": [1, 1, 1, 2, 2, 3, 3, 3],
 "wave": [1, 2, 3, 1, 3, 1, 2, 3],
 "mood": [4.0, 3.5, 3.0, 2.0, 2.5, 5.0, 4.5, 4.0],
 "sleep": [7.0, 6.5, 6.0, 5.5, 5.0, 8.0, 7.5, 7.0],
 "age": [60, 60, 60, 55, 55, 52, 52, 52],
 }
)
```

## Step 2: Pivot into wide format and keep missing waves as `NaN`

```python
wide_df = (
 long_df
 .pivot(index="subject_id", columns="wave", values=["mood", "sleep"])
 .sort_index(axis=1)
)
wide_df.columns = [f"{feature}_w{wave}" for feature, wave in wide_df.columns]
wide_df = wide_df.reset_index().merge(
 long_df[["subject_id", "age"]].drop_duplicates(), on="subject_id"
)

print(wide_df)
```

Output (note the `NaN` values for missing waves):

```
 subject_id mood_w1 mood_w2 mood_w3 sleep_w1 sleep_w2 sleep_w3 age
0 1 4.0 3.5 3.0 7.0 6.5 6.0 60
1 2 2.0 NaN 2.5 5.5 NaN 5.0 55
2 3 5.0 4.5 4.0 8.0 7.5 7.0 52
```

Wide format lets you keep all subject data on a single row while leaving missing visits as `NaN`.

!!! note "Why wide format?"
    `Sklong` expects one row per subject to avoid leakage during train/test splits. Missing visits should stay as
    `NaN` in the wave columns so they can be imputed later if needed.

## Step 3: Define `features_group` with padding for missing **columns**

If **a whole wave column is missing for a feature**, use `-1` to align its group length with the maximum wave count.
In the example below, `sleep_w2` is missing entirely (no column), while `mood_w2` exists but is `NaN` for subject 2.

```python
from scikit_longitudinal.data_preparation import LongitudinalDataset

columns = [
 "age",
 "gender",
 "mood_w1",
 "mood_w2",
 "mood_w3",
 "sleep_w1",
 "sleep_w3", # sleep_w2 does not exist in this dataset
 "outcome_w3",
]

# Example with uneven visits + missing sleep_w2 column
wide_dataset = pd.DataFrame(
 [
 [60, 0, 4.0, 3.5, 3.0, 7.0, 6.0, 1],
 [55, 1, 2.0, None, 2.5, 5.5, 5.0, 0],
 [52, 0, 5.0, 4.5, 4.0, 8.0, 7.0, 1],
 ],
 columns=columns,
)

# Create the dataset (data_frame skips file IO)
longitudinal = LongitudinalDataset(file_path="unused.csv", data_frame=wide_dataset)
longitudinal.load_target(target_column="outcome_w3")

# features_group uses -1 to pad the missing sleep wave
features_group = [
 [2, 3, 4], # mood_w1, mood_w2, mood_w3
 [5, -1, 6], # sleep_w1, N/A, sleep_w3
]
longitudinal.setup_features_group(features_group)

print(longitudinal.feature_groups(names=True))
print(longitudinal.non_longitudinal_features(names=True))
```

!!! warning "Padding is for missing columns, not missing values"
    Use `-1` only when an entire wave column does not exist in your data. If a wave exists but is missing for some
    subjects, keep the column and use `NaN` values for those rows. For instance, sleep has not been measured at wave 2 of
    the longitudinal study, so we use `-1` to pad that position in `features_group`. As a result, we now that sleep measured
    at wave 3, is way more recent than sleep measured at wave 1. given the distance between the two.

Output:

```
[["mood_w1", "mood_w2", "mood_w3"], ["sleep_w1", "N/A", "sleep_w3"]]
["age", "gender", "outcome_w3"]
```

This setup correctly informs `Sklong` about the temporal structure of your data, even with uneven observations.
Usually, most of Sklong current primitives handles missing waves (NaN values) by taking the distance between the waves into account.
Hence, they are "aware" of whether one wave is way more recent than another.

Cheers!
