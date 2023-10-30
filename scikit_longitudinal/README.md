<!--suppress HtmlDeprecatedAttribute -->
<div align="center">
   <p align="center">
   <h1 align="center">
      <br>
      <a href="./../logo.png"><img src="./../logo.png" alt="Scikit-longitudinal" width="200"></a>
      <br>
      Scikit-longitudinal Key Features & Contributions
      <br>
   </h1>
   <h4 align="center">A specialised Python library for longitudinal data analysis built on Scikit-learn</h4>
</div>

> 🌟 **Exciting Update**: We're delighted to introduce the brand new v0.1 documentation for Scikit-longitudinal! For a deep dive into the library's capabilities and features, please [visit here](https://simonprovost.github.io/scikit-longitudinal/).


> ⚠️ **DISCLAIMER**: This README pertains specifically to the primary features of the library. For a comprehensive
> introduction to the library, including its setup, please refer to the [main readme](./../README.md). Furthermore,
> this README is intended for developers contributing to the library.

## ⭐️Key Features

### 📈 Classifier estimators

|             Key Feature              |                               Location in Code                                |
|:------------------------------------:|:-----------------------------------------------------------------------------:|
|    **Nested Trees Classifier** 🌲    |         [View Code](./estimators/trees/nested_trees/nested_trees.py)          |
| **Lexicographical Random Forest** 🌳 | [View Code](./estimators/trees/lexicographical_trees/lexico_random_forest.py) |
| **Lexicographical Decision Tree** 🌲 | [View Code](./estimators/trees/lexicographical_trees/lexico_decision_tree.py) |

### 🚀📉 Pre Processing Estimators

|                               Key Feature                                |                                      Location in Code                                       |
|:------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|
| **Correlation-based Feature Selection Per Group (CFS-PerGroup v1 & v2)** | [View Code](preprocessors/feature_selection/correlation_feature_selection/cfs_per_group.py) |
|              **Correlation-based Feature Selection (CFS)**               |      [View Code](preprocessors/feature_selection/correlation_feature_selection/cfs.py)      |

### 🛠️ Additional Tools and Metrics

|                     Key Feature                      |      Location in Code      |
|:----------------------------------------------------:|:--------------------------:|
|         **Scikit-Longitudinal Pipeline** 🔧          | [View Code](./pipeline.py) |
| **Area Under The Recall-Precision Curve (AUPRC)** 📉 | [View Code](./metrics.py)  |

## 🤝 Contributing (developers)

### Temporal Structures Unveiled: Essential Insights and Illustrations 🧪

When designing machine learning algorithms for longitudinal data within the Scikit-longitudinal library, an algorithm
must gains access to crucial features and mechanisms, such as how recent a feature is relative to the same feature at a
different time point, or whether a feature is a longitudinal attribute or a non-longitudinal attribute.

By introducing two important attributes, Scikit-Longitudinal has been meticulously designed to facilitate the management
and interpretation of the temporal structure of data, thereby empowering algorithm designers to craft models that are
not only adaptable but also generalisable. These attributes are:

1. **Features Group (`features_group: List[List[int]]`):**
    - `features_group` consist of a list of integer lists. Each inner list encapsulates a set of (indexes) features
      corresponding to a particular longitudinal attribute, with the order of indices within these lists representing
      the sequence of waves (time points), thereby preserving the crucial temporal dependencies for algorithms focusing
      on longitudinal data.

      In the absence of data for a particular wave (i.e, missing wave for a given longitudinal attribute), a placeholder
      value of `-1` is used to maintain the same temporal dependencies across all inner lists (or across all
      longitudinal attributes). With the correct configuration of this attribute, machine learning models are enabled to
      leverage the data's
      temporal structure, thereby capturing the intricate patterns unique to longitudinal datasets in a manner shared by
      all longitudinal machine learning
      algorithms.

2. **Non-longitudinal Features (`non_longitudinal_features: List[int]`):**
    - On the other hand, the `non_longitudinal_features` attribute provides a list of feature indices devoid of temporal
      characteristics. While these features could remain potentially unaffected by the algorithm's longitudinal nature,
      they
      serve to enhance the data's context and informational value by complementing the longitudinal features. For
      instance, the capacity to determine whether a given feature is longitudinal or not. Utilising this
      attribute guarantees an exhaustive comprehension of the dataset's temporal and non-temporal dimensions.

Consequently, the combination of these two characteristics enables algorithm designers to develop models that are not
only adaptable but also generalisable and reusable. E.g., Instead of hardcoding the temporal structures of the
designer's dataset or even more
complex, relying on feature names (hard-coded). Utilising these characteristics provides a dynamic method for maximising
the potential of longitudinal data and machine learning algorithm adapted to longitudinal data.

Last but not least, the `-1` missing waves information is helpful for understanding how recent a given feature is given
that all inner list are the same lenghts, but if
your proposed technique is not interested in this specific, then the `clean_padding` function plays a vital role in this
regard by removing any padding from the `features_group`, thereby preventing interference. This function ensures that
only pertinent features are considered, potentially and ideally improving the model's precision and efficacy.

### Example of a Longitudinal Dataset: A Visual Representation

Given a longitudinal dataset, let's denote the number of waves (time points) as $`N`$. Each wave represents a specific
time point of data capture. The feature set for a specific longitudinal variable across all waves can be represented as:

```math
\mathcal{F}_x = \{ \text{Feature}_{x_{\text{wave}_1}}, \text{Feature}_{x_{\text{wave}_2}}, \ldots, \text{Feature}_{x_
{\text{wave}_N}} \}
```

Using this notation, the `features_group` for the dataset can be constructed by collecting all such sets for each
longitudinal variable. Suppose the dataset contains $`M`$ longitudinal variables. Then, the `features_group` can be
represented as:

```math
\mathcal{G} = \{ \mathcal{F}_1, \mathcal{F}_2, \ldots, \mathcal{F}_M \}
```

Where each feature set $`\mathcal{F}_x`$, where $`x`$ goes from $`1 \ldots M`$, is a list of features corresponding to a particular longitudinal variable,
and $`\text{Feature}_{x_{\text{wave}_i}}`$ is the feature for that longitudinal variable at wave $`i`$. Where $`i`$ goes
from $`1 \ldots N`$, $`N`$ being the total number of waves. Hence, $`\mathcal{G}`$ being what we denote as `features group`.

Next, let's provide a visual representation of this structure:

#### Mock Dataset Example

Below is a representation of a mock dataset:

| Patient ID | Blood Pressure (Wave 1) | Blood Pressure (Wave 2) | Blood Pressure (Wave 3) | Heart Rate (Wave 1) | Heart Rate (Wave 3) | Age | Gender |
|------------|-------------------------|-------------------------|-------------------------|---------------------|---------------------|-----|--------|
| 1          | 120                     | 125                     | 122                     | 75                  | 73                  | 45  | M      |
| 2          | 115                     | 118                     | 119                     | 78                  | 74                  | 50  | F      |
| 3          | 123                     | 127                     | 125                     | 74                  | 76                  | 52  | M      |

Where each row represents a different patient. Therefore, some columns delineate characteristics for each wave, while others delineate static characteristics. 
For the mock exemple, it is organised so that similar characteristics across waves are placed adjacently for clarity. However, it is essential to note that in
practice, features could be arranged in any order. In addition, observe that _Heart Rate (Wave 2)_ is absent for every
patient.

Consequently, by scrutinising this structure, one may straightforwardly derive the `features_group`
and `non_longitudinal_features`
for this dataset:

```python
features_group = [
    ['Blood Pressure (Wave 1)', 'Blood Pressure (Wave 2)', 'Blood Pressure (Wave 3)'],
    ['Heart Rate (Wave 1)', 'N/A', 'Heart Rate (Wave 3)']  # N/A is utilised as a placeholder for missing waves.
]
non_longitudinal_features = ['Age', 'Gender']

# However, in practice, the features_group and non_longitudinal_features are represented with the indices of the features. Such that:

features_group = [[1, 2, 3], [4, -1, 5]]  # -1 is utilised as a placeholder for missing waves.
non_longitudinal_features = [6, 7]
```

Upon undergoing `clean_padding`, the `features_group` would appear as follows:

```python
features_group = [[1, 2, 3], [4, 5]]  # -1 is extricated.
```

Consequently, these two attributes are provided to the algorithm, allowing it to leverage the data's temporal structure and non-temporal structure in a consistent manner across all longitudinal algorithms. Finally, to generate the content of these two attributes based on your dataset, you will have to use the Longitudinal Dataset's class to load your CSV with: [View Code](https://github.com/simonprovost/scikit-longitudinal/blob/main/scikit_longitudinal/data_preparation/longitudinal_dataset.py)

____

### Process how to contribute a New Algorithm to the library 🔬

1. **Fork & Clone**: Start by forking this repository and then clone your forked repository locally.
2. **Environment Setup**: Ensure you've set up the development environment correctly. Install necessary dependencies as
   mentioned in the [main readme](./../README.md).
3. **Develop the Algorithm**:
    - Determine the phase of the ML workflow your algorithm belongs to (e.g., Classifier, Pre-processing).
    - Follow the module structure and place your code in the appropriate directory.
    - Make sure your algorithm adheres to Scikit-learn's API conventions. Take use of our custom templates in the
      [templates](./templates/__init__.py) folder.
4. **Testing**: Write unit tests for your algorithm to ensure it functions correctly, especially when handling
   longitudinal data.
5. **Documentation**: Provide clear documentation and usage examples.
6. **Pull Request**: Create a pull request from your forked repository to the main repository detailing your new feature
   and any other relevant information. Reviewers will then review your pull request and provide feedback for it
   to be potentially merged into the main branch.
