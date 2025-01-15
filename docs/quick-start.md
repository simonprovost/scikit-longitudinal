---
hide:
  - navigation
---

# 💡 About The Project

Longitudinal datasets contain information about the same cohort of individuals (instances) over time, 
with the same set of features (variables) repeatedly measured across different time points 
(also called `waves`) [1,2,3].

`Scikit-longitudinal` (Sklong) is a machine learning library designed to analyse
longitudinal data, also called _Panel data_ in certain fields. Today, Sklong focuses on Longitudinal Machine Learning Classification tasks.
It offers tools and models for processing, analysing, 
and classifying longitudinal data, with a user-friendly interface that 
integrates with the `Scikit-learn` ecosystem.

For further information, visit the [official documentation](https://simonprovost.github.io/scikit-longitudinal/).

---

## 🛠️ Installation

### Step 1: Install the Latest Version of Sklong

```shell
pip install scikit-longitudinal
```

To install a specific version, specify it explicitly:
```shell
pip install scikit-longitudinal==0.0.6
```
Refer to the [Release Notes](https://github.com/simonprovost/scikit-longitudinal/releases) for available versions.

---

### Installing `Scikit-longitudinal` as Part of a Project

#### 🫵 Project Setup: Using PDM

If you’re managing your project dependencies with `PDM`, note that `Scikit-longitudinal` is a fork of `Scikit-Learn` and is incompatible with the original `Scikit-Learn` package. To ensure compatibility, exclude `Scikit-Learn` from your project dependencies by adding the following configuration to your `pyproject.toml` file:

````toml
[tool.pdm.resolution]
excludes = ["scikit-learn"]
````

This ensures that the modified version of `Scikit-Learn`—`Scikit-Lexicographical-Trees`—is used seamlessly within your project.

To install dependencies:
```shell
pdm install
```

To install only production dependencies:
```shell
pdm install --prod
```

For additional configurations, refer to the [PDM documentation](https://pdm.fming.dev/).

---

#### 🫵 Project Setup: Using UV

If you prefer **UV** for dependency management, configure your `pyproject.toml` file to override conflicting packages. Add the following configuration:

````toml
[tool.uv]
package = true
override-dependencies = [
    "scikit-learn ; sys_platform == 'never'",
]
````

Steps to set up your environment:
1. **Create a Virtual Environment:**
   ```bash
   uv venv
   ```

2. **Pin the Required Python Version:**
   ```bash
   uv python pin cpython-3.9.21
   ```

3. **Lock Dependencies:**
   ```bash
   uv lock
   ```

4. **Install All Dependencies:**
   ```bash
   uv sync --all-groups
   ```

5. **Run Tests:**
   ```bash
   uv run pytest -sv scikit_longitudinal
   ```

For more information, refer to the [UV documentation](https://docs.astral.sh/uv/).

---

#### 🐾 Installing `Scikit-longitudinal` on Apple Silicon Macs

Apple Silicon-based Macs require running under an `x86_64` architecture to ensure proper installation and functioning of `Scikit-longitudinal`. This is primarily due to the `Deep-Forest` dependency being incompatible with Apple Silicon.

Note below we us **UV** yet you can use any other virtual environment manager of interest as long as you are able to switch between architectures (see step 1).

**Steps to Configure:**

1. **Start a Terminal Session Under `x86_64` Architecture**:
   ```bash
   arch -x86_64 zsh
   ```

2. **Install an `x86_64` Compatible Python Version with UV**:
   ```bash
   uv python install cpython-3.9.21-macos-x86_64-none
   uv python pin cpython-3.9.21-macos-x86_64-none
   ```

3. **Create and Activate a Virtual Environment**:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

4. **Install `Scikit-longitudinal`**:
   ```bash
   uv pip install scikit-longitudinal
   ```

5. **Run Tests**:
   ```bash
   uv run pytest scikit_longitudinal/ --cov=./ --cov-report=html --cov-config=.coveragerc --cov-report=html:htmlcov/scikit_longitudinal -s -vv --capture=no
   ```

Refer to [UV documentation](https://docs.astral.sh/uv/) for further details.

---

### 💻 Developer Notes

For developers looking to contribute, please refer to the `Contributing` section of the [documentation](https://simonprovost.github.io/scikit-longitudinal/).

---

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

dataset = LongitudinalDataset('./stroke.csv') # Note this is a fictional dataset. Use yours!
dataset.load_data_target_train_test_split(
  target_column="class_stroke_wave_4",
)

# Pre-set or manually set your temporal dependencies 
dataset.setup_features_group(input_data="elsa")

model = LexicoGradientBoostingClassifier(
  features_group=dataset.feature_groups(),
  threshold_gain=0.00015 # Refer to the API for more hyper-parameters and their meaning
)

model.fit(dataset.X_train, dataset.y_train)
y_pred = model.predict(dataset.X_test)

# Classification report
print(classification_report(y_test, y_pred))
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
of Ageing. Artificial Intelligence Review, 57(4), p.84ibeiro, C. and Freitas, A.A., 2019. A mini-survey of supervised machine learning approaches for coping with ageing-related longitudinal datasets. In 3rd Workshop on AI for Aging, Rehabilitation and Independent Assisted Living (ARIAL), held as part of IJCAI-2019 (num. of pages: 5).