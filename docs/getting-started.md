---
hide:
  - navigation
---

# 💡 `Scikit-longitudinal`, in a nutshell!

Biomedical research often uses longitudinal data with repeated measurements of variables across time (e.g. cholesterol measured across time), which is challenging for standard machine learning algorithms due to intrinsic temporal dependencies.

`Scikit-longitudinal` (abbreviated `Sklong`, pronounced /ˌɛs keɪ ˈlɒŋ/ or "Ess-kay-long" and /ˌsaɪ kɪt ˌlɒndʒɪˈtjuːdɪnəl/ or "Sky-kit lon-ji-TOO-din-ul") is a machine learning library for longitudinal machine learning classification tasks. We focus on supervised learning—currently classification—and provide two complementary ways to work with longitudinal datasets:

- **Data preparation**: flatten or otherwise restructure longitudinal tables into static, tabular representations to plug into standard machine learning workflows.
- **Algorithm adaptation**: keep temporal dependencies intact and train longitudinal-aware estimators that leverage the temporal structure.

You will find a gentle introduction to these paths throughout the [tutorials](tutorials/index.md), while this page offers the essentials to get started and points you toward the right deeper dives.

Note that while Longitudinal datasets have a temporal component, other types of datasets, such as time series,
also have a temporal component but are not considered longitudinal datasets. Time series data typically involves
a single variable measured at regular intervals over time, while longitudinal datasets involve multiple variables
measured across the same cohort of individuals at different time points. More is discussed in the [FAQ](faq.md).
However, I would like to highlight that time points are therefore considered as `waves` in `Sklong` [^1][^2][^3].

!!! example "Explore more about Scikit-Longitudinal within the paper"
    The library is presented in [Scikit-Longitudinal: A Machine Learning Library for Longitudinal Classification in Python](https://doi.org/10.21105/joss.08481), published in the *Journal of Open Source Software (JOSS)*.
    If you want a concise overview of the design decisions and capabilities, start there before diving into the examples below.

To start your Longitudinal Machine Learning journey with `Sklong`, you first will have to install the library.

---

## 🛠️ Installation

!!! warning "Operating System & Python Support"
    `Scikit-longitudinal` is supported on Python `3.10`–`3.13` for `Ubuntu` (or other stable Linux distributions) and `macOS`.
    On `Windows`, please use `Google Colab` or a `Docker` image based on `Ubuntu` because one dependency does not currently
    ship compatible wheels.

Choose the installation method that best suits your workflow. For more troubleshooting tips, see the [Developers page](developers.md#%EF%B8%8F-installation--troubleshooting-first).

=== "UV (Recommended)"

    We recommend using `UV`, a fast and efficient Python package manager that simplifies environment and dependency management. Install `UV` if you haven't already. See [UV installation instructions](https://docs.astral.sh/uv/).

    Add `Scikit-longitudinal` to an existing UV-managed project:

    ```bash
    uv add scikit-longitudinal
    ```

    Extras work the same way:

    ```bash
    uv add "scikit-longitudinal[parallelisation]"
    uv add "scikit-longitudinal[dev]"
    ```

    You can also pin a specific version if desired:

    ```bash
    uv add "scikit-longitudinal==0.0.8"
    ```

    After adding, run `uv sync` to materialise the lockfile.

=== ":simple-pypi: Pip (Standard)"

    Install the latest release from PyPI:

    ```bash
    pip install scikit-longitudinal
    ```

    Extras are available for specialised needs:

    ```bash
    pip install "scikit-longitudinal[parallelisation]"  # Ray-based parallel workloads
    pip install "scikit-longitudinal[dev]"              # Docs, testing, linting
    ```

    You can also pin a specific version if desired:

    ```bash
    pip install scikit-longitudinal==0.0.8
    ```

=== ":simple-python: Conda (CondaForge)"

    To install `Scikit-longitudinal` using `Conda`, follow these steps:

    1. Open your terminal or Anaconda Prompt.
    2. Create a new Conda environment with Python 3.10:

       ```bash
       conda create --name sklong -c conda-forge python=3.10
       ```

    3. Activate the environment:

       ```bash
       conda activate sklong
       ```

    4. Install `Scikit-longitudinal`:

       ```bash
       pip install scikit-longitudinal
       ```

    You can also pin a specific version if desired:

    ```bash
    pip install scikit-longitudinal==0.0.8
    ```

    This will install `Scikit-longitudinal` in your newly created Conda environment.

=== ":simple-jupyter: Jupyter with UV (~1 line)"

    Launch `Jupyter Lab` with `Sklong` in a temporary environment managed by `UV`:

    ```bash
    uv run --with scikit_longitudinal jupyter lab
    ```

    This uses your default Python (3.10–3.13). To pin a specific Python, pass `--python <path_or_version>`.

=== ":simple-googlecolab: Google Colab (~4 lines)"

    1. Open a new Colab notebook (Python 3.10+).
    2. Install `Sklong`:

       ```bash
       !pip install scikit-longitudinal
       ```

    3. Ensure the compatible `scikit-lexicographical-trees` dependency is present:

       ```bash
       !pip install scikit-lexicographical-trees
       ```

    4. Remove conflicting `scikit-learn` if preinstalled:

       ```bash
       !pip uninstall scikit-learn -y
       ```

=== ":light_blue_heart: Marimo"

    !!! warning
        Support for Marimo is incoming. If you're interested in contributing to this feature, please submit a pull request!

=== ":simple-codingninjas: Within your PyProject"

    If you are setting up `Sklong` inside a project, ensure your dependency manager prefers
    `scikit-lexicographical-trees` instead of `scikit-learn` (which is incompatible with Sklong) **and** declare `scikit-longitudinal` in your project metadata.

    #### 🫵 Project Setup: Using PDM

    Add the dependency and exclude `scikit-learn` in `pyproject.toml`:

    ````toml
    [project]
    dependencies = [
        "scikit-longitudinal",
    ]

    [tool.pdm.resolution]
    excludes = ["scikit-learn"]
    ````

    Install dependencies:
    ```shell
    pdm install
    ```

    #### 🫵 Project Setup: Using UV

    Declare the dependency and override the incompatible wheel:

    ````toml
    [project]
    dependencies = [
        "scikit-longitudinal",
    ]

    [tool.uv]
    package = true
    override-dependencies = [
        "scikit-learn ; sys_platform == 'never'",
    ]
    ````

    Then sync your environment:

    ```bash
    uv sync --all-groups
    ```

!!! note "Have trouble installing Sklong?"
    Known issues and workarounds live in the [installation & troubleshooting](developers.md#%EF%B8%8F-installation--troubleshooting-first) section of the Developers guide.

---

## 🚀 Your first Longitudinal-data aware Classification Task

`Sklong` has numerous primitives to deal with longitudinal machine learning classification tasks. To begin, use the
[`LongitudinalDataset`](API/data_preparation/longitudinal_dataset.md) class to prepare your dataset, including `data` and `temporal vectors`. The example below shows the
algorithm-adaptation path via the longitudinal-aware [`LexicoGradientBoostingClassifier`](API/estimators/ensemble/lexico_gradient_boosting.md); for data-preparation (flattening)
workflows, hop into the [tutorials](tutorials/index.md) for alternative pipelines.

Here's our basic workflow example:

```python
from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_gradient_boosting import
    LexicoGradientBoostingClassifier
from sklearn.metrics import classification_report

# Load your dataset (replace 'stroke.csv' with your actual dataset path)
dataset = LongitudinalDataset('./stroke.csv')

# Set up the target column and split the data into training and testing sets (replace 'stroke_wave_4' with your target column)
dataset.load_data_target_train_test_split(
    target_column="class_stroke_wave_4",
)

# Set up feature groups (temporal dependencies)
# Use a pre-set for English Longitudinal Study of Ageing (ELSA) data or define manually
# (see the tutorials for defining custom temporal dependency groups)
dataset.setup_features_group(input_data="elsa")

# Initialise the classifier with feature groups
model = LexicoGradientBoostingClassifier(
    features_group=dataset.feature_groups(),
    threshold_gain=0.00015  # Adjust hyperparameters as needed (See further in the API reference)
)

# Fit the model to the training data
model.fit(dataset.X_train, dataset.y_train)

# Make predictions on the test data
y_pred = model.predict(dataset.X_test)

# Print the classification report
print(classification_report(dataset.y_test, y_pred))
```

!!! info "What is the LexicoGradientBoostingClassifier?"
    A variant of [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
    tailored for longitudinal data, using a lexicographical approach that prioritises recent `waves` over older ones in
    certain scenarios[^3].

!!! tip "Where's the data at?"
    The `scikit-longitudinal` library does not include datasets by default, primarily for privacy reasons.
    You can use your own longitudinal datasets or download publicly available ones, such as the [ELSA dataset](https://www.elsa-project.ac.uk/).
    Interested in synthetic datasets? [Open an issue](https://github.com/simonprovost/scikit-longitudinal/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen), and we can explore it together.

---

## What's Next?

Start with the [tutorials](tutorials/index.md) to see hands-on examples of data preparation versus algorithm adaptation
in action. Then visit the [community hub](community-hub.md) to connect with other users and contributors.

Advanced users can dive into the [API](API/index.md) for full details on estimators, primitives, and configuration
options.
___

[^1]: Kelloway, E.K. and Francis, L., 2012. Longitudinal research and data analysis. In Research methods in occupational
health psychology (pp. 374-394). Routledge.

[^2]: Ribeiro, C. and Freitas, A.A., 2019. A mini-survey of supervised machine learning approaches for coping with
ageing-related longitudinal datasets. In 3rd Workshop on AI for Aging, Rehabilitation and Independent Assisted Living (
ARIAL), held as part of IJCAI-2019 (num. of pages: 5).

[^3]: Ribeiro, C. and Freitas, A.A., 2024. A lexicographic optimisation approach to promote more recent features on
longitudinal decision-tree-based classifiers: applications to the English Longitudinal Study of Ageing. Artificial
Intelligence Review, 57(4), p.84.

