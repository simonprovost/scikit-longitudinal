---
hide:
  - navigation
---

# 💡 About The Project

Longitudinal datasets contain information about the same cohort of individuals (instances) over time, 
with the same set of features (variables) repeatedly measured across different time points 
(also called `waves`) [1,2].

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

## 🚀 Getting Started

To use `Sklong`, start by preparing your dataset using the `LongitudinalDataset` class, and then train a model with tools like `LexicoGradientBoostingClassifier`.

Here’s a quick example:

````python
from sklearn.metrics import classification_report
from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.estimators.ensemble.lexicographical import LexicoGradientBoostingClassifier

# Prepare the dataset
dataset = LongitudinalDataset('./stroke.csv')
dataset.load_data_target_train_test_split(target_column="class_stroke_wave_4")
dataset.setup_features_group(input_data="Elsa")

# Train the classifier
model = LexicoGradientBoostingClassifier(features_group=dataset.feature_groups(), threshold_gain=0.00015)
model.fit(dataset.X_train, dataset.y_train)

# Evaluate the model
y_pred = model.predict(dataset.X_test)
print(classification_report(dataset.y_test, y_pred))
````

For more examples, visit the [Examples](https://simonprovost.github.io/scikit-longitudinal/examples) section of the documentation.

---

# 📚 References

> [1] Kelloway, E.K. and Francis, L., 2012. Longitudinal research and data analysis. In Research methods in occupational health psychology (pp. 374-394). Routledge.

> [2] Ribeiro, C. and Freitas, A.A., 2019. A mini-survey of supervised machine learning approaches for coping with ageing-related longitudinal datasets. In 3rd Workshop on AI for Aging, Rehabilitation and Independent Assisted Living (ARIAL), held as part of IJCAI-2019 (num. of pages: 5).