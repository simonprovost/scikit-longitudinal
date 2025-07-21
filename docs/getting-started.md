---
hide:
  - navigation
---

# 💡 `Scikit-longitudinal`, in a nutshell!

# 💡 `Scikit-longitudinal`, in a nutshell!

Biomedical research often uses longitudinal data with repeated measurements of variables across time (e.g. cholesterol
measured across time), which is challenging for standard machine learning algorithms due to intrinsic temporal
dependencies. 

`Scikit-longitudinal` (abbreviated `Sklong`, pronounced /ˌɛs keɪ ˈlɒŋ/ or "Ess-kay-long" and /ˌsaɪ kɪt
ˌlɒndʒɪˈtjuːdɪnəl/ or "Sky-kit lon-ji-TOO-din-ul") is a machine learning library helping out in anlysing longitudinal 
data, also known as _panel data_ in some fields. `Sklong` specialises in longitudinal machine learning _classification_ tasks,
offering user-friendly tools for `processing`, `analyzing`, and `classifying` longitudinal tabular data, seamlessly
integrating with the `Scikit-learn` ecosystem.

Note that while Longitudinal datasets have a temporal component, other types of datasets, such as time series,
also have a temporal component but are not considered longitudinal datasets. Time series data typically involves
a single variable measured at regular intervals over time, while longitudinal datasets involve multiple variables
measured across the same cohort of individuals at different time points. More is discussed in the [FAQ](https://scikit-longitudinal.readthedocs.io/latest//faq/).
However, I would like to highlight that time points are therefore considered as `waves` in `Sklong` [^1][^2][^3].

To start your Longitudinal Machine Learning journey with `Sklong`, you first will have to install the library.

---

## 🛠️ Installation

!!! warning "Operating System Support"
    `Scikit-longitudinal` is currently supported on `OSX` (MacOS) and `Linux`. 
    `Windows` users should use `notebooks` or `Docker` with a `Linux` (
    Ubuntu) distribution due to limitations with a dependency library. For more details, open an issue, I would be 
    happy to discuss this out further.

!!! important "Python Version Compatibility"
    `Scikit-longitudinal` is currently compatible with Python versions `3.9` only.
    Ensure you have one of these versions installed before proceeding with the installation.

    Now, while we understand that this is a limitation, we are tied for the time being because of `Deep Forest`.
    `Deep Forest` is a dependency of `Scikit-longitudinal` that is not compatible with Python versions greater than `3.9`.
    `Deep Forest` helps us with the `Deep Forest` algorithm, to which we have made some modifications to
    welcome `Lexicographical Deep Forest`.

    To follow up on this discussion, please refer to [this github issue](https://github.com/LAMDA-NJU/Deep-Forest/issues/124).

Please, start by choosing the installation method that best suits your needs:

=== ":simple-pypi: PyPi (Classic Install)"

    To install `Scikit-longitudinal`, you can use `pip`:

    ```bash
    pip install scikit-longitudinal
    ```

    This will install the latest version of `Scikit-longitudinal` from the Python Package Index (PyPI).

    If you want to install a specific version, you can specify it like this:

    ```bash
    pip install scikit-longitudinal==0.0.8  # Replace with the desired version
    ```

    Please note that here we assume you have a compatible Python version installed (3.9) and a working environment (e.g Conda).

=== ":simple-python: Conda (CondaForge)"

    To install `Scikit-longitudinal` using `Conda`, follow these steps:

    1. Open your terminal or Anaconda Prompt.
    2. Create a new Conda environment with Python 3.9:

       ```bash
       conda create --name sklong -c conda-forge python=3.9 
       ```

    3. Activate the environment:

       ```bash
       conda activate sklong
       ```

    4. Install `Scikit-longitudinal`:

       ```bash
       pip install scikit-longitudinal
       ```

    This will install `Scikit-longitudinal` in your newly created Conda environment.

=== ":simple-jupyter: Jupyter Notebook (~ 1 line) via `UV`"

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
        uv python pin 3.9
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
    [gist](https://gist.github.com/simonprovost/356030bd8f1ea077bdbc120fdc116c16#file-support_39_scikit_longitudinal_in_google_colab-ipynb) 
    –– or –– [Open in Google Colab :simple-googlecolab:](tutorials/temporal_dependency.md){ .md-button }
    
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

    5. Remove & Re-Install `Scikit-lexicographical-trees`, which is the modified version of `Scikit-learn` used by `Scikit-longitudinal`:

       ```bash
       !pip uninstall scikit-lexicographical-trees -y
       !pip install scikit-lexicographical-trees
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
    
    ### 💻 Developer Notes
    
    For developers looking to contribute, please refer to the `Contributing` section of the [documentation](https://scikit-longitudinal.readthedocs.io/latest//).

---

## 🚀 Quick Start (Code)

`Sklong` has numerous primitives to deal with longitudinal machine learning classification tasks. To begin, use the
`LongitudinalDataset` class to prepare your dataset, including `data` and `temporal vectors`. To train a machine learning
classifier on your data, use estimators such as `LexicoGradientBoostingClassifier`.

> "The `LexicoGradientBoostingClassifier` in a nutshell: a variant
> of [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
> tailored for longitudinal data, using a lexicographical approach that prioritises recent `waves` over older ones in
> certain scenarios[^3]."

!!! tip "Where's the data at?"
    The `scikit-longitudinal` library does not include datasets by default. Mainly due to the privacy reason.
    You can use your own longitudinal datasets
    or download publicly available ones, such as the [ELSA dataset](https://www.elsa-project.ac.uk/). 
    If synthetic datasets is of interest, open an issue, I would be happy to discuss this out further.

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

!!! question "How do I tune hyperparameters?"
    Check the [API Reference](https://scikit-longitudinal.readthedocs.io/latest/API/) for a complete list of
    hyperparameters and their meanings.

!!! question "Neural Network Models?"
    Scikit-longitudinal currently does not support neural network-based models. For similar projects that do, see
    the [FAQ](https://scikit-longitudinal.readthedocs.io/latest//faq/) section.

---

## What's Next?

Next, we highly recommend that you explore the `Temporal Dependency` section, which provides a comprehensive
understanding of how to set up temporal dependencies in your longitudinal datasets. This is crucial for
building effective longitudinal machine learning models with `Scikit-longitudinal`.

Following that? You could play it all by walking through the `API Reference` section, which provides detailed
information on the various estimators, primitives, and hyperparameters available in `Scikit-longitudinal`.
___

[^1]: Kelloway, E.K. and Francis, L., 2012. Longitudinal research and data analysis. In Research methods in occupational
health psychology (pp. 374-394). Routledge.

[^2]: Ribeiro, C. and Freitas, A.A., 2019. A mini-survey of supervised machine learning approaches for coping with
ageing-related longitudinal datasets. In 3rd Workshop on AI for Aging, Rehabilitation and Independent Assisted Living (
ARIAL), held as part of IJCAI-2019 (num. of pages: 5).

[^3]: Ribeiro, C. and Freitas, A.A., 2024. A lexicographic optimisation approach to promote more recent features on
longitudinal decision-tree-based classifiers: applications to the English Longitudinal Study of Ageing. Artificial
Intelligence Review, 57(4), p.84.


## 🚨 Troubleshooting

=== ":simple-apple: Install On Apple Silicon Chips"
    Apple Silicon-based Macs require running under an `x86_64` architecture to ensure proper installation and
    functioning of `Scikit-longitudinal`. This is primarily due to the `Deep-Forest` dependency being incompatible
    with Apple Silicon (ref Github Issue with `Deep Forest`
    authors [here](https://github.com/LAMDA-NJU/Deep-Forest/issues/133)).

    The following steps are somehow extracted & adapted
    from [https://apple.stackexchange.com/a/408379](https://apple.stackexchange.com/a/408379).

    1. **Install Rosetta 2**
       Rosetta 2 allows your Apple Silicon Mac (M1, M2, etc.) to run apps built for the `x86_64` architecture,
       which is needed for `Scikit-longitudinal` due to its `Deep-Forest` dependency.
        ```bash
        softwareupdate --install-rosetta
        ```
       Note: When prompted, press `'A'` and `Enter` to agree to the license terms.
    
    2. **Launch Terminal in `x86_64` Mode**
       Restart your terminal (close and reopen it), then start a new shell session under the `x86_64` architecture.
       ```bash
       arch -x86_64 zsh
       ```
       Note: To verify so you can run `uname -m` and it should output `x86_64`.
    
    3. **Install `Scikit-longitudinal` via `Conda` with `Pip`**
       ```bash
       conda create --name sklong python=3.9
       conda activate sklong
       pip install scikit-longitudinal
       ```
    4. **Verify Installation**
       ```bash
       python -c "import scikit_longitudinal"
       ```
    
    And voila! You should now have `Scikit-longitudinal` installed and running on your Apple Silicon Mac.