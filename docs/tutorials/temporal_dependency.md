---
---


# ⏳ Understanding Temporal Dependencies in Longitudinal Datasets
## ⏳ Understanding Temporal Dependencies in Longitudinal Datasets

In the following, we will look at the only component of `Sklong` that is completely novel compared to previous
libraries. Our explanation covers how we generalise the use of temporal dependencies in longitudinal datasets across 
all `Sklong` algorithms, methods, and primitives.

This reading aims to help you establish up longitudinal datasets in 'Sklong' that allow you to utilise any algorithm
without worrying about temporal dependencies. The primitives will handle this based on their design.

!!! quote "Common Shared Objects"

    The overall goal is to establish a general method for representing the temporal dependency of longitudinal data. 
    Although hardcoding feature names in a freshly constructed method may work for a specific dataset, 
    it cannot be generalised to other datasets, or not easily though.

    We therefore introduce:

    - `features_group`
    - `non_longitudinal_features`
    
    These objects are intended to be integrated into any algorithm for longitudinal data classification within 
    `Scikit-Longitudinal`. Hence, by correctly structuring these objects, algorithm designers can take use of 
    longitudinal data's temporal structure without e.g. requiring considerable hardcoding and be able to wider the range
    of potential users of their algorithms.

    The following sections will explain how to configure these objects for your datasets.


!!! tip "Representing Longitudinal Data Linearly —— Wide format instead of Long format"

    Instead of representing the same subject across multiple rows—where, for example, row two for a subject 
    represents wave 2 for each features, and row three represents wave 3, we represent the same subject across columns. In this format, the column names define the wave/time-point for the data features of each subject.

    This representation may initially seem confusing, but it is the most common method for tabular longitudinal data
    due to several advantages. Two could be:

    1. **Prevention of Data Leakage**: When performing cross-validation, segmenting data in the middle of a subject's 
    record (when represented in rows) can lead to data leakage. Representing subjects in rows with time-point 
    features as columns prevents this issue.
    2. **Simplified Data Interpretation**: Each row represents a single subject, eliminating the need to cross-reference
    multiple rows to understand a subject's progression over time, thus reducing cognitive load. Furthermore, non-longitudinal features,
    in this format, are easily identifiable as they are not repeated across columns, whereas in the row format, they would be despite not changing over time, which could lead to redundancy in the data.

    If your current dataset represents subjects across rows, you should pivot your data to have each row represent a 
    subject with features over time in columns. If this pivot becomes a frequent need, we plan to offer a tool 
    within `LongitudinalDataset` to automate this process. Please open an issue if you require this feature!

## Understanding `features_group`

`features_group` is a list of lists of integers, with each inner list representing a group of features for a specific 
longitudinal variable. The inner lists' indices are ordered by wave/time-point sequence, capturing the
temporal dependencies required for longitudinal data algorithms.

Consider a dataset with four features, two of which are longitudinal and each consist of two records collected over time,
called waves or time-points. A real-world example would be `smoke` and `cholesterol`, each with two 
waves/time-points, and you want to divide them into two groups, one with the first longitudinal attribute, 
`smoke`, which is made up of the two feature indices in the dataset about `smoke`, and the other with the second l
ongitudinal attribute, `cholesterol`. In this case, you would pass the following list of lists of integers as 
the `features_group` parameter:

``` py
[[0,1],[2,3]]
```

Here, `0` and `1` are the indices of the first longitudinal attribute `smoke`, and `2` and `3` are the 
indices of the second longitudinal attribute `cholesterol`. So `0` is `smoke` wave/time-point `1`, `1` is 
`smoke` wave/time-point `2`, `2` is `cholesterol` wave/time-point `1`, and `3` is `cholesterol` wave/time-point 
`2`. Hence, the algorithm can deal with the feature recentness, i.e., the first element of the inner 
lists are older, and the farther the element is from the first element, the more recent it is.

## Understanding `non_longitudinal_features`

`non_longitudinal_features` contains indices for non-temporal features. These features have no temporal order 
and can be handled separately by the algorithms. However, how these features are treated is determined by 
the algorithm designer. This means that algorithm designers can or cannot incorporate these features 
into their algorithms, watch out for the algorithm's documentation to know how it handles parameters.

To come back to the object. For example, if you have a dataset with 5 features, where the first 4 
are longitudinal attributes (`features_group` as `[[0,1],[2,3]]`), and the last one is non-longitudinal, 
you would pass the following list of integers in the `non_longitudinal_features` parameter:

``` py
[4]
```

!!! Tip "An example of Non Longitudinal Features"
    In the case of a dataset with longitudinal features such as `smoke` and `cholesterol`, 
    non-longitudinal features could be `age` and `gender`, because they are typically not collected over time. E.g,
    for `age` the evolution is `normal` so despite being collected at different time-points, it does not add much value.
    Nonetheless, in certain scenario these features could be longitudinal, depends on the task at hand.

## Let's take an exemplary dataset

!!! example "Consider the following dummy-dataset: `stroke_longitudinal.csv`"

    ??? note "Dummy Dataset"
        This dataset is a dummy dataset for the sake of the example. It is not real data and should not be expected
        for any real-world analysis. Please, use your own longitudinal dataset for your analysis throughout any
        of the examples available in this whole documentation.
    
    Features:

    - `smoke` (longitudinal) with two waves/time-points
    - `cholesterol` (longitudinal) with two waves/time-points
    - `blood_pressure` (longitudinal) with two waves/time-points
    - `diabetes` (longitudinal) with two waves/time-points
    - `exercise` (longitudinal) with two waves/time-points
    - `obesity` (longitudinal) with two waves/time-points
    - `age` (non-longitudinal)
    - `gender` (non-longitudinal)

    The dataset is shown below (`w` stands for `wave` in ELSA):

    | age | gender | smoke_w1 | smoke_w2 | cholesterol_w1 | cholesterol_w2 | blood_pressure_w1 | blood_pressure_w2 | diabetes_w1 | diabetes_w2 | exercise_w1 | exercise_w2 | obesity_w1 | obesity_w2 | stroke_w2 |
    |-----|--------|----------|----------|----------------|----------------|-------------------|-------------------|-------------|-------------|-------------|-------------|------------|------------|-----------|
    | 66  | 0      | 0        | 1        | 0              | 0              | 0                 | 0                 | 1           | 1           | 0           | 1           | 0          | 0          | 0         |
    | 59  | 0      | 0        | 0        | 1              | 1              | 0                 | 0                 | 1           | 1           | 1           | 1           | 1          | 1          | 1         |
    | 63  | 0      | 0        | 0        | 1              | 1              | 0                 | 0                 | 0           | 0           | 0           | 0           | 0          | 0          | 1         |
    | 47  | 0      | 0        | 0        | 1              | 1              | 0                 | 0                 | 0           | 0           | 0           | 0           | 1          | 0          | 0         |
    | 44  | 0      | 0        | 0        | 1              | 1              | 1                 | 1                 | 0           | 0           | 0           | 0           | 1          | 1          | 1         |
    | 69  | 1      | 0        | 0        | 0              | 0              | 1                 | 1                 | 0           | 0           | 0           | 0           | 0          | 0          | 0         |
    | 63  | 0      | 0        | 0        | 0              | 0              | 0                 | 0                 | 0           | 0           | 0           | 0           | 0          | 0          | 0         |
    | 48  | 1      | 0        | 0        | 0              | 0              | 0                 | 0                 | 0           | 0           | 0           | 1           | 0          | 0          | 0         |
    | 49  | 1      | 0        | 0        | 0              | 0              | 0                 | 0                 | 0           | 0           | 0           | 1           | 0          | 1          | 0         |

Now let's set up the `features_group` and `non_longitudinal_features` for this dataset for Sklong:

``` py
from scikit_longitudinal.data_preparation import LongitudinalDataset

dataset = LongitudinalDataset('./stroke_longitudinal.csv')
dataset.load_data()

# Manually set your temporal dependencies
dataset.setup_features_group(
    input_data=[[2,3],[4,5],[6,7],[8,9],[10,11],[12,13]],
)

print(f"Features group: {dataset.feature_groups(names=True)}")
>$ Features group: [['smoke_w1', 'smoke_w2'], ['cholesterol_w1', 'cholesterol_w2'], ['blood_pressure_w1', 'blood_pressure_w2'], ['diabetes_w1', 'diabetes_w2'], ['exercise_w1', 'exercise_w2'], ['obesity_w1', 'obesity_w2']]
print(f"Non-longitudinal features: {dataset.non_longitudinal_features(names=True)}")
>$ Non-longitudinal features: ['age', 'gender']
```

# Pre-set `features_group` and `non_longitudinal_features`

Presets are configurations that can be used to quickly set up the `features_group` and `non_longitudinal_features`
based on a given popular database.

We currently have a pre-set configuration for the `features_group` and `non_longitudinal_features` in 
the [English Longitudinal Study of Ageing (ELSA)](https://www.elsa-project.ac.uk/) database. The `ELSA` database is an ageing-related diseases 
longitudinal database that can be accessed via this link: [ELSA](https://www.elsa-project.ac.uk/). 

> The `ELSA` database tracks core participants, who are 50 years of age or older and reside in the United Kingdom, 
> through repeated interviews. For instance, biomedical data collected every four years by a nurse or health
> professional results in ELSA-nurse datasets, while data from core interviews conducted every two years results
> in ELSA-core datasets.

Instead of using your own configuration for the `input_data` parameter of the `setup_features_group` 
method of `LongitudinalDataset`, you can use the pre-set configuration for the 
`ELSA` database, which is passed as a string to the 
`input_data` parameter. It will generate the `features_group` and `non_longitudinal_features` 
for you based on how the data is constructed. An exemplary usage is shown below:

``` py
from scikit_longitudinal.data_preparation import LongitudinalDataset

dataset = LongitudinalDataset('./extended_stroke_longitudinal.csv')
dataset.load_data()

# Pre-set your temporal dependencies
dataset.setup_features_group(input_data="elsa")

print(f"Features group: {dataset.feature_groups(names=True)}")
>$ ...will print the features group of the ELSA dataset ...
print(f"Non-longitudinal features: {dataset.non_longitudinal_features(names=True)}")
>$ ...will print the non-longitudinal features of the ELSA dataset ...
```

!!! success "More Presets, stay tuned!"
    More presets may appear in the future; contribute yours if you believe they will benefit the community! If more
    than one pre-set configuration is available, we will open a new section in the 
    `API Reference` to list them all.


To conclude, the appropriate configuration of `features_group` and `non_longitudinal_features` is critical 
for algorithm designers and library users. It allows anyone to use the temporal structure of 
longitudinal data as well as non-temporal features to capture the underlying patterns in the data using 
any adapted/newly designed algorithms for longitudinal data classification tasks in a **shared-common** 
and **user-friendly** manner. As a result, instead of having to create an algorithm that only works on one
dataset by, for example, hard-coding the dataset's temporal structure, another one could, 
for example, rely on the feature names, rendering it inapplicable to other datasets.