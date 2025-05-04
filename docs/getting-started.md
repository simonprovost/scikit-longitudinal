---
hide:
  - navigation
---

## 💡 Getting Started with `Scikit-longitudinal`

Longitudinal datasets contain information about the same cohort of individuals (instances) over time, with the same set
of features (variables) repeatedly measured across different time points (also called `waves`)[^1][^2][^3].

`Scikit-longitudinal` (Sklong) is a machine learning library designed to analyse longitudinal data, also known as _Panel
data_ in certain fields. Today, Sklong focuses on Longitudinal Machine Learning _Classification_ tasks. It offers tools
and models for `processing`, `analysing`, and `classifying` longitudinal tabular data, with a user-friendly 
interface that integrates with the `Scikit-learn` ecosystem.

Let's first jump into the installation, and then we will explore the first steps with code and some common questions.
We then suggest to explore the API reference and the examples to get a better understanding of the library.

---

## 🛠️ Installation

!!! warning "Operating System Support"
    `Scikit-longitudinal` is currently supported on `OSX` (MacOS) and `Linux`. `Windows` users should use `notebooks` or `Docker` with a `Linux` (
    Ubuntu) distribution due to limitations with a dependency library. For more details, see
    the [Scikit-Lexicographical-Trees GitHub repository](https://github.com/simonprovost/scikit-lexicographical-trees). Feel
    free to contribute a Windows-based wheel to unlock this potential!

### Installation Methods

Please, start by choosing the installation method that best suits your needs:

=== ":simple-jupyter: Jupyter Notebook (~ 1 line)"

    To run `Jupyter lab` with `Scikit-longitudinal`, we recommend using `UV`, a fast and efficient Python package 
    manager that simplifies environment and dependency management.

    Here's how to set it up:

    1. Install `UV` if you haven't already. See [UV installation instructions](https://docs.astral.sh/uv/).
    2. Run the following command to launch `Jupyter lab` with `Scikit-longitudinal`:

       ```bash
       uv run --python /usr/bin/python3 --with scikit_longitudinal jupyter lab
       ```

       Replace `/usr/bin/python3` with the path to your desired Python version, as long as it is `3.9` and less than `3.10`, it has been tested.
       For more options, refer to the [UV CLI documentation](https://docs.astral.sh/uv/reference/cli/#uv-python).

    You are ready to play with `Scikit-longitudinal` in `Jupyter lab`! 🎉

    ??? question "How to install different version if we do not have `3.9`?"
        You can install a different version of Python using `uv` by running:

        ```bash
        uv python install 3.9
        ```
        This command will install Python 3.9 and set it as the default version for your environment.

    ??? question "How do I get the path to my just installed Python version?"
        You can find the path to your installed Python version by running:

        ```bash
        uv python list --all-versions
        ```
        This command will list all installed Python versions along with their paths. If a path is present, it means
        that the version is installed. You can then use the path in the `uv run` command.

    This command creates a temporary environment with `Scikit-longitudinal` installed and starts `Jupyter lab`.

    ––––––––––––––

    Some shoutouts to the `UV` team for their amazing work! 🙌

    ![UV Proof](https://github.com/astral-sh/uv/assets/1309177/03aa9163-1c79-4a87-a31d-7a9311ed9310#only-dark)

    !!! tip "UV's readings recommendations:"
        - [Python Packaging in Rust](https://astral.sh/blog/uv)
        - [A Year of UV](https://www.bitecode.dev/p/a-year-of-uv-pros-cons-and-should)
        - [UV Is All You Need](https://dev.to/astrojuanlu/python-packaging-is-great-now-uv-is-all-you-need-4i2d)
        - [State of the Art Python 2024](https://4zm.org/2024/10/28/state-of-the-art-python-in-2024.html)
        - [Data Scientist, From School to Work](https://towardsdatascience.com/data-scientist-from-school-to-work-part-i/)


=== ":simple-googlecolab: Google Colab (~5 lines)"

    To use `Scikit-longitudinal` in `Google Colab`, follow these steps due to compatibility requirements:
    
    You also can follow the follwing gist as we reproduce the below's steps: 
    [gist](https://gist.github.com/simonprovost/356030bd8f1ea077bdbc120fdc116c16#file-support_39_scikit_longitudinal_in_google_colab-ipynb) –– or –– [Open in Google Colab :simple-googlecolab:](https://scikit-longitudinal.readthedocs.io/latest//temporal_dependency/){ .md-button }
    
    Preliminary steps:
    
    1. Open a new `Google Colab` notebook.
    2. Open in a code / text editor the notebook you want to use `Scikit-longitudinal` in, and proceed with the following modifications.
       
        ```
        "metadata": {
            ...
            "kernelspec": {
              "name": "python3.9",
              "display_name": "Python 3.9"
            },
            ...
        },
        ```
    3. Save the notebook.

    You are ready to add stuff in your notebook!

    1. Downgrade the Python version to 3.9, as `Scikit-longitudinal` supports this version.
         You can do this by running the following command in a code cell:

         Shoutout to @J3soon for [this solution](https://github.com/j3soon/colab-python-version)!
    
         ```bash
         !wget -O py39.sh https://raw.githubusercontent.com/j3soon/colab-python-version/main/scripts/py39.sh
         !bash py39.sh
         ```
    
         This command installs Python 3.9 on your Colab instance.

    2. It'll automatically refresh the kernel to ensure it uses Python 3.9, no worries!
    3. Install `Scikit-longitudinal`:

       ```bash
       !pip install scikit-longitudinal
       ```

    4. Remove `Scikit-learn` if installed, as `Scikit-longitudinal` is a fork and may conflict:

       ```bash
       !pip uninstall scikit-learn -y
       ```

    After these steps, you can use `Scikit-longitudinal` in your Colab notebook 🎉

=== ":light_blue_heart: Marimo"

    Support for Marimo is incoming. If you're interested in contributing to this feature, please submit a pull request!

=== ":simple-codingninjas: Within A Project"

    For developing your own scripts with Scikit-longitudinal, install it via pip:

    ```bash
    pip install scikit-longitudinal
    ```

    Following the pip install, let's explore various scarions. As follows:

    #### 🫵 Project Setup: Using PDM
    
    If you’re managing your project dependencies with `PDM`, note that `Scikit-longitudinal` is a fork of `Scikit-Learn` 
    and is incompatible with the original `Scikit-Learn` package. To ensure compatibility, exclude `Scikit-Learn` from 
    your project dependencies by adding the following configuration to your `pyproject.toml` file:
    
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
    
    If you prefer **UV** for dependency management, configure your `pyproject.toml` file to override conflicting 
    packages. Add the following configuration:
    
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
       uv python pin cpython-3.9.21 # Or any other version you want as long as it fits Sklon requiements.
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
    
    #### 🐾 Installing `Scikit-longitudinal` on Apple Silicon Macs – Troubleshooting
    
    Apple Silicon-based Macs require running under an `x86_64` architecture to ensure proper installation and
    functioning of `Scikit-longitudinal`. This is primarily due to the `Deep-Forest` dependency being incompatible 
    with Apple Silicon.
    
    Note below we us **UV** yet you can use any other virtual environment manager of interest as long as you are able 
    to switch between architectures (see step 1).
    
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
       uv pip install scikit-longitudinal # Can try uv add scikit-longitudinal at this point!
       ```
    
    5. **Run Tests**:
       ```bash
       uv run pytest scikit_longitudinal/ --cov=./ --cov-report=html --cov-config=.coveragerc --cov-report=html:htmlcov/scikit_longitudinal -s -vv --capture=no
       ```
    
    Refer to [UV documentation](https://docs.astral.sh/uv/) for further details.
    
    ---
    
    ### 💻 Developer Notes
    
    For developers looking to contribute, please refer to the `Contributing` section of the [documentation](https://scikit-longitudinal.readthedocs.io/latest//).


---

## 🚀 First Steps with Code

To perform longitudinal machine learning classification with `Sklong`, use the `LongitudinalDataset` class to prepare
your dataset (data, temporal vectors, etc.). Then, analyse your data with estimators like
`LexicoGradientBoostingClassifier`.

> "The `LexicoGradientBoostingClassifier` in a nutshell: a variant
> of [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
> tailored for longitudinal data, using a lexicographical approach that prioritises recent `waves` over older ones in
> certain scenarios[^3]."

Here's a basic example:

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
# Use a pre-set for ELSA data or define manually (See further in the API reference)
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

---

## ❓ Questions

!!! question "What are feature groups?"
    Feature groups define the temporal dependencies in your longitudinal data. They are lists of feature indices
    corresponding to different waves. See
    the [Temporal Dependency](https://scikit-longitudinal.readthedocs.io/latest//temporal-dependency/) section for more
    details.

!!! question "How do I set temporal dependencies?"
    Use pre-sets for known datasets like ELSA or define them manually based on your data structure. Refer to
    the [Temporal Dependency](https://scikit-longitudinal.readthedocs.io/latest//temporal-dependency/) section.

!!! question "Where can I find more examples?"
    Explore the [Examples](https://scikit-longitudinal.readthedocs.io/latest//examples/) section for additional use cases
    and code snippets.

!!! question "How do I tune hyperparameters?"
    Check the [API Reference](https://scikit-longitudinal.readthedocs.io/latest/API/) for a complete list of
    hyperparameters and their meanings.

!!! warning "Neural Network Models"
    Scikit-longitudinal currently does not support neural network-based models. For similar projects that do, see
    the [FAQ](https://scikit-longitudinal.readthedocs.io/latest//faq/) section.

---

[^1]: Kelloway, E.K. and Francis, L., 2012. Longitudinal research and data analysis. In Research methods in occupational
health psychology (pp. 374-394). Routledge.

[^2]: Ribeiro, C. and Freitas, A.A., 2019. A mini-survey of supervised machine learning approaches for coping with
ageing-related longitudinal datasets. In 3rd Workshop on AI for Aging, Rehabilitation and Independent Assisted Living (
ARIAL), held as part of IJCAI-2019 (num. of pages: 5).

[^3]: Ribeiro, C. and Freitas, A.A., 2024. A lexicographic optimisation approach to promote more recent features on
longitudinal decision-tree-based classifiers: applications to the English Longitudinal Study of Ageing. Artificial
Intelligence Review, 57(4), p.84.