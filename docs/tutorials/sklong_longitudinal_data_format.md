---
icon: lucide/table-properties
---

# Longitudinal Data Format

!!! tip "Dataset Used in Tutorials"
    Use the shared synthetic dataset defined in the [tutorials overview](overview.md#dataset-used-in-tutorials). Generate it once there and reuse it here.

This tutorial introduces the data formats supported by Scikit-Longitudinal (`Sklong`), focusing on the wide format for longitudinal data. We'll explain why wide format is preferred to avoid data leakage and provide examples using a simple synthetic dataset.

## Wide vs. Long Format

The animation below contrasts the two layouts at split time: a long-format split can cut through a single subject's rows and leak wave-level information across train/test, while a wide-format split always keeps one subject on one side.

<figure class="expandable-media" markdown="span" style="text-align: center;">
 [![Wide vs. long format: leakage at split time](../assets/images/tutorials/sklong_longitudinal_data_format/WideVsLongLeakage.avif#only-light){ width="100%" loading="lazy" }](../assets/images/tutorials/sklong_longitudinal_data_format/WideVsLongLeakage.avif){ .expandable-media__trigger }
 [![Wide vs. long format: leakage at split time](../assets/images/tutorials/sklong_longitudinal_data_format/WideVsLongLeakageDark.avif#only-dark){ width="100%" loading="lazy" }](../assets/images/tutorials/sklong_longitudinal_data_format/WideVsLongLeakageDark.avif){ .expandable-media__trigger }
 <figcaption>Click the image to expand it.</figcaption>
</figure>

Longitudinal data can be represented in two main formats:

- **Long Format**: Each observation for a subject is in a separate row, with a time indicator column. This can lead to data leakage during splitting if rows for the same subject are separated.

- **Wide Format**: Each subject is in one row, with columns for each feature at each time point (e.g., `smoke_w1`, `smoke_w2`). This format prevents leakage as all temporal data for a subject stays together.

`Sklong` focuses on wide format for safety and simplicity in machine learning workflows.

## Synthetic Dataset Example

The dataset used is synthetic, mimicking health data for illustration.

- Longitudinal features: `smoke`, `cholesterol`, `blood_pressure`, `diabetes`, `exercise`, `obesity` (two waves each).
- Non-longitudinal: `age`, `gender`.
- Target: `stroke_w2` (binary).

Load and prepare:

```python
from scikit_longitudinal.data_preparation import LongitudinalDataset

dataset = LongitudinalDataset('./extended_stroke_longitudinal.csv')
dataset.load_data()
dataset.load_target(target_column='stroke_w2')
dataset.setup_features_group([[2,3], [4,5], [6,7], [8,9], [10,11], [12,13]]) # Indices for longitudinal features
dataset.load_train_test_split(test_size=0.2, random_state=42)

print(dataset.data.head())
```

This wide format ensures safe splitting and temporal integrity.

!!! tip "Need to convert from long to wide?"
    See the dedicated [Long ⇄ Wide Reshape](long_wide_reshape.md) tutorial for a step-by-step walkthrough — including the recommended pandas pivot, handling uneven waves, and going back the other way.
