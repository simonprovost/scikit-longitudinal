# Tutorials Overview :books:

The following will guide you through the understanding of Scikit-Longitudinal (`Sklong`) through a series of tutorials.
We aim at offering an understanding first to what is a Longitudinal dataset per our definition, how do we generalise its temporal
dependencies, what shape (wide/long) do we expect your data to be in, and how to use the estimators and pipelines
available in `Sklong`.

<figure markdown="span" style="text-align: center;">
  ![Scikit-Longitudinal Banner](../assets/images/SklongBanner.png){ width="100%" loading="lazy" }
</figure>

_Click on the image to enlarge it._

The above figure illustrates the general flow of `Scikit-Longitudinal` from raw longitudinal data to the various primitives available.
As follows.

First, we would like to give you an overview of `Scikit-Longitudinal`, from raw longitudinal data to the various available
primitives. We begin by presenting the raw longitudinal (**wide-format**) data. We distinguish the temporal dependencies (
waves or time points) using colours (or focus the arrows). While the raw data simply requests that you provide data in a wide format, we then
demonstrate that you must prepare this data (via the `Data preparation` step) using either a `data transformation` or `algorithm adaptation` approach. Data transformation reshapes the longitudinal table into a static tabular representation suitable for any `Scikit-learn` compatible estimator. Algorithm adaptation keeps the temporal dependencies intact so you can train the longitudinal-aware primitives discussed in the [JOSS paper](https://doi.org/10.21105/joss.08481) and the [API reference](../API/index.md).

Before all, recall that Scikit-Longitudinal is a library that extends the `Scikit-learn` ecosystem to handle longitudinal data.
That is, if you do not know about `Scikit-learn`, we recommend you first read the
[Scikit-learn documentation](https://scikit-learn.org/latest/user_guide.html) to understand the popular Fit—Predict—Transform API.

## :books: Overview of Tutorials

<div class="grid cards" markdown>

-   :fontawesome-solid-timeline:{ .lg .middle } __Temporal Dependency__

    ---

    Learn how to set up temporal dependencies using `features_group` and `non_longitudinal_features`. Essential for all
    `Sklong` usage.

    [:octicons-arrow-right-24: Read the tutorial](temporal_dependency.md)

-   :material-table:{ .lg .middle } __Longitudinal Data Format__

    ---

    Understand wide vs. long formats and why `Sklong` prefers wide. Includes loading and preparing data.

    [:octicons-arrow-right-24: Read the tutorial](sklong_longitudinal_data_format.md)

-   :material-database-cog:{ .lg .middle } __Data Preparation: Flatten Temporal Dependency for Scikit-Learn Estimators__

    ---

    Flatten longitudinal structure and plug it into standard estimators using transformations like `AggrFunc`.

    [:octicons-arrow-right-24: Read the tutorial](sklong_data_preparation_first_exploration.md)

-   :octicons-pulse-24:{ .lg .middle } __Algorithm Adaptation: Preserve Temporal Dependency for Sklong Estimators__

    ---

    Fit and predict with a longitudinal-aware estimator like `LexicoDecisionTreeClassifier`.

    [:octicons-arrow-right-24: Read the tutorial](sklong_explore_your_first_estimator.md)

-   :material-pipe:{ .lg .middle } __Pipelines: Mix Longitudinal Components__

    ---

    Build a full pipeline combining transformation, preprocessing, and estimation steps.

    [:octicons-arrow-right-24: Read the tutorial](sklong_explore_your_first_pipeline.md)

-   :material-tune:{ .lg .middle } __Hyperparameter Tuning: Grid vs. Random Search__

    ---

    Compare grid search and random search for tuning `LexicoRandomForestClassifier` hyperparameters.

    [:octicons-arrow-right-24: Read the tutorial](sklong_hyperparameter_tuning.md)

-   :material-robot-happy-outline:{ .lg .middle } __Automated Machine Learning (CASH)__

    ---

    Automate model selection and tuning across pipelines with Auto-Sklong.

    [:octicons-rocket-16: Jump onto Auto-Sklong](https://github.com/simonprovost/Auto-Sklong)

</div>

!!! important "Dataset Used in Tutorials"
    All tutorials share the same synthetic health-inspired longitudinal dataset. Generate it once and reuse it across lessons:

    ```python
    import pandas as pd
    import numpy as np

    n_rows = 500
    columns = [
        'age', 'gender',
        'smoke_w1', 'smoke_w2',
        'cholesterol_w1', 'cholesterol_w2',
        'blood_pressure_w1', 'blood_pressure_w2',
        'diabetes_w1', 'diabetes_w2',
        'exercise_w1', 'exercise_w2',
        'obesity_w1', 'obesity_w2',
        'stroke_w2'
    ]

    data = []
    for _ in range(n_rows):
        row = {
            'age': np.random.randint(40, 71),
            'gender': np.random.choice([0, 1]),
        }
        for feature in ['smoke', 'cholesterol', 'blood_pressure', 'diabetes', 'exercise', 'obesity']:
            w1 = np.random.choice([0, 1], p=[0.7, 0.3])
            w2 = np.random.choice([0, 1], p=[0.2, 0.8]) if w1 == 1 else np.random.choice([0, 1], p=[0.9, 0.1])
            row[f'{feature}_w1'] = w1
            row[f'{feature}_w2'] = w2

        stroke_risk = row['smoke_w2'] == 1 or row['cholesterol_w2'] == 1 or row['blood_pressure_w2'] == 1
        p_stroke = 0.2 if stroke_risk else 0.05
        row['stroke_w2'] = np.random.choice([0, 1], p=[1 - p_stroke, p_stroke])
        data.append(row)

    df = pd.DataFrame(data)
    csv_file = './extended_stroke_longitudinal.csv'
    df.to_csv(csv_file, index=False)
    print(f"Extended CSV file '{csv_file}' created successfully.")
    ```

    | age | gender | smoke_w1 | smoke_w2 | cholesterol_w1 | cholesterol_w2 | blood_pressure_w1 | blood_pressure_w2 | diabetes_w1 | diabetes_w2 | exercise_w1 | exercise_w2 | obesity_w1 | obesity_w2 | stroke_w2 |
    |-----|--------|----------|----------|----------------|----------------|-------------------|-------------------|-------------|-------------|-------------|-------------|------------|------------|-----------|
    | 66  | 0      | 0        | 1        | 0              | 0              | 0                 | 0                 | 1           | 1           | 0           | 1           | 0          | 0          | 0         |
    | 59  | 0      | 0        | 0        | 1              | 1              | 0                 | 0                 | 1           | 1           | 1           | 1           | 1          | 1          | 1         |
    | 63  | 0      | 0        | 0        | 1              | 1              | 0                 | 0                 | 0           | 0           | 0           | 0           | 0          | 0          | 1         |

!!! tip "Prerequisite Reading"
    To get the most from each tutorial, start with the [Temporal Dependency guide](temporal_dependency.md) and the [Longitudinal Data Format walkthrough](sklong_longitudinal_data_format.md). These cover the core concepts every example relies on.