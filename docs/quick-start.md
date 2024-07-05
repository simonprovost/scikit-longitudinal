---
hide:
  - navigation
---

# 💡 About The Project
# 💡 About The Project

Longitudinal datasets contain information about the same cohort of individuals (instances) over time, 
with the same set of features (variables) repeatedly measured across different time points 
(also called `waves`) [1,2].

`Scikit-longitudinal` (Sklong) is a machine learning library designed to analyse
longitudinal data, also called _Panel data_ in certain fields. Today, Sklong is focussed on the Longitudinal Machine Learning Classification task.
It offers tools and models for processing, analysing, 
and classify longitudinal data, with a user-friendly interface that 
integrates with the `Scikit-learn` ecosystem.

For further information, visit the [official documentation](https://simonprovost.github.io/scikit-longitudinal/).

## 🛠️ Installation

To install `Sklong`, follow these two easy steps:

1. ✅ **Install the latest version of `Sklong`**:

    ```shell
    pip install Scikit-longitudinal
    ```
    !!! info "Different Versions?"
        You can also install different versions of the library by specifying the version number, e.g., `pip install Scikit-longitudinal==0.0.1`. 
        Refer to the [Release Notes](https://github.com/simonprovost/scikit-longitudinal/releases).

2. 📦 **[MANDATORY] Update the required dependencies**

    !!! info "Why is this necessary?"
        See [this explanation](https://github.com/pdm-project/pdm/issues/1316#issuecomment-2106457708).

    `Scikit-longitudinal` includes a modified version of `Scikit-Learn` called `Scikit-Lexicographical-Trees`, which can be found at [this Pypi link](https://pypi.org/project/scikit-lexicographical-trees/).

    This revised version ensures compatibility with the unique features of `Scikit-longitudinal`. However, conflicts may occur with other dependencies that also require `Scikit-Learn`. Follow these steps to prevent any issues when running your project.

    <details>
    <summary><strong>🫵 Simple Setup: Command Line Installation</strong></summary>

    If you want to try `Sklong` in a simple environment without a proper `pyproject.toml` file (such as using `Poetry`, `PDM`, etc.), run the following command:

    ```shell
    pip uninstall scikit-learn && pip install scikit-lexicographical-trees
    ```
    </details>

    <details>
    <summary><strong>🫵 Project Setup: Using `PDM` (or any other package manager such as `Poetry`, etc.)</strong></summary>

    If you have a project managed by `PDM`, or any other package manager, the example below demonstrates `PDM`. The process is similar for `Poetry` and others. Consult their documentation for instructions on excluding a package.

    To prevent dependency conflicts, you can exclude `Scikit-Learn` by adding the following configuration to your `pyproject.toml` file:

    ```toml
    [tool.pdm.resolution]
    excludes = ["scikit-learn"]
    ```

    *This exclusion ensures `Scikit-Lexicographical-Trees` (used as `Scikit-Learn`) is used seamlessly within your project.*
    </details>

### 💻 Developer Notes

For developers looking to contribute, please refer to the `Contributing` section of the [documentation](https://simonprovost.github.io/scikit-longitudinal/).

## 🛠️ Supported Operating Systems

`Scikit-longitudinal` is compatible with the following operating systems:

- MacOS  
- Linux 🐧
- Windows via Docker only (Docker uses Linux containers) 🪟 

!!! warning
    We haven't tested it on Windows without Docker.

## 🚀 Getting Started

To perform longitudinal machine learning classification using `Sklong`, start by employing the
`LongitudinalDataset` class to prepare your dataset (i.e, data itself, temporal vector, etc.). To analyse your data, 
you can utilise for instance the `LexicoGradientBoostingClassifier` or any other available estimator/preprocessor. 

> "The `LexicoGradientBoostingClassifier` in a nutshell: is a variant of 
> [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
> specifically designed for longitudinal data, using a lexicographical approach that prioritises recent
> `waves` over older ones in certain scenarios [1].

Next, you can apply the popular _fit_, _predict_, _prodict_proba_, or _transform_
methods depending on what you previously employed in the same way that `Scikit-learn` does, as shown in the example below:

``` py
from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_gradient_boosting import LexicoGradientBoostingClassifier

dataset = LongitudinalDataset('./stroke.csv')
dataset.load_data_target_train_test_split(
  target_column="class_stroke_wave_4",
)

# Pre-set or manually set your temporal dependencies 
dataset.setup_features_group(input_data="Elsa")

model = LexicoGradientBoostingClassifier(
  features_group=dataset.feature_groups(),
  threshold_gain=0.00015 # Refer to the API for more hyper-parameters and their meaning
)

model.fit(dataset.X_train, dataset.y_train)
y_pred = model.predict(dataset.X_test)
```

!!! warning "Neural Networks models"
    Please see the documentation's `FAQ` tab for a list of similar projects that may offer 
    Neural Network-based models, as this project presently does not. 
    If we are interested in building Neural Network-based models for longitudinal data, 
    we will announce it in due course.

!!! question "Wants to understand what's the feature_groups? How your temporal dependencies are set via pre-set or manually?"
    To understand how to set your temporal dependencies, please refer to the `Temporal Dependency` tab of the documentation.

!!! question "Wants more to grasp the idea?"
    To see more examples, please refer to the `Examples` tab of the documentation.

!!! question "Wants more control on hyper-parameters?"
    To see the full API reference, please refer to the `API` tab.

# 📚 References

> [1] Kelloway, E.K. and Francis, L., 2012. Longitudinal research and data analysis. In Research methods in occupational health psychology (pp. 374-394). Routledge.

> [2] Ribeiro, C. and Freitas, A.A., 2019. A mini-survey of supervised machine learning approaches for coping with ageing-related longitudinal datasets. In 3rd Workshop on AI for Aging, Rehabilitation and Independent Assisted Living (ARIAL), held as part of IJCAI-2019 (num. of pages: 5).

> [3] Ribeiro, C. and Freitas, A.A., 2024. A lexicographic optimisation approach to promote more recent 
features on longitudinal decision-tree-based classifiers: applications to the English Longitudinal Study 
of Ageing. Artificial Intelligence Review, 57(4), p.84