---
icon: lucide/play
---

# Get started

Biomedical research often uses longitudinal data with repeated measurements of variables across time, which is challenging for standard machine learning algorithms due to intrinsic temporal dependencies.

`Scikit-longitudinal` (abbreviated `Sklong`, pronounced /ˌɛs keɪ ˈlɒŋ/ or "Ess-kay-long" and /ˌsaɪ kɪt ˌlɒndʒɪˈtjuːdɪnəl/ or "Sky-kit lon-ji-TOO-din-ul") is a machine learning library for longitudinal classification workflows. It provides two complementary ways to work with repeated-measures data:

- **Data preparation**: flatten or restructure longitudinal tables into static, tabular representations for standard machine learning workflows.
- **Algorithm adaptation**: preserve temporal dependencies and train longitudinal-aware estimators that leverage the wave structure directly.

Time series and longitudinal datasets both involve a temporal component, but they are not the same. Time series data typically follows one variable over time, while longitudinal datasets follow multiple variables across the same cohort of individuals at different time points. In `Sklong`, those time points are treated as `waves` [^1][^2][^3].

!!! example "Read the official paper"
    If you would like the architectural view before the hands-on one, start there and then come back to the installation steps below. The library is presented in [Scikit-Longitudinal: A Machine Learning Library for Longitudinal Classification in Python](https://doi.org/10.21105/joss.08481), published in the *Journal of Open Source Software (JOSS)*.

## Installation

Choose the installation method that best suits your workflow.

!!! info "Operating system and Python support"
    Use Python `3.10` to `3.13`.
    Native support is available on `macOS` and stable Linux distributions such as `Ubuntu`.
    On `Windows`, use `Google Colab` or a Linux-based `Docker` image for now.

=== "UV <span class='tab-badge tab-badge--accent'>Recommended</span>"

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
    uv add "scikit-longitudinal==0.1.8"
    ```

    After adding, run `uv sync` to materialise the lockfile.

    !!! tip "Need UV first?"
        See the [UV installation instructions](https://docs.astral.sh/uv/).

=== "Pip <span class='tab-badge'>Standard</span>"

    Install the latest release from PyPI:

    ```bash
    pip install scikit-longitudinal
    ```

    Extras are available for specialised needs:

    ```bash
    pip install "scikit-longitudinal[parallelisation]" # Ray-based parallel workloads
    pip install "scikit-longitudinal[dev]" # Docs, testing, linting
    ```

    You can also pin a specific version if desired:

    ```bash
    pip install scikit-longitudinal==0.1.8
    ```

=== "Conda <span class='tab-badge'>CondaForge</span>"

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
    pip install scikit-longitudinal==0.1.8
    ```

    This installs `Scikit-longitudinal` in your newly created Conda environment.

=== "Jupyter with UV <span class='tab-badge'>1 line</span>"

    Launch `Jupyter Lab` with `Sklong` in a temporary environment managed by `UV`:

    ```bash
    uv run --with scikit_longitudinal jupyter lab
    ```

    This uses your default Python (`3.10` to `3.13`). To pin a specific Python, pass `--python <path_or_version>`.

=== "Google Colab <span class='tab-badge'>4 lines</span>"

    1. Open a new Colab notebook (Python 3.10–3.13).
    2. Remove the preinstalled stock `scikit-learn` (Sklong relies on the `scikit-lexicographical-trees` fork which ships its own `sklearn` package):

    ```bash
    !pip uninstall scikit-learn -y
    ```

    3. Install `Sklong` (this pulls in `scikit-lexicographical-trees` automatically):

    ```bash
    !pip install scikit-longitudinal
    ```

=== "Marimo"

    !!! info
        Support for Marimo is incoming. If you are interested in contributing to this feature, please submit a pull request.

=== "Within your PyProject"

    If you are setting up `Sklong` inside a project, ensure your dependency manager prefers `scikit-lexicographical-trees` instead of `scikit-learn` and declare `scikit-longitudinal` in your project metadata.

    **Project setup with PDM**

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

    **Project setup with UV**

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

---

!!! bug "Have trouble installing Sklong?"
    Known issues and workarounds live in the [installation and troubleshooting](../developers.md#installation-troubleshooting) section of the Developers guide.

[^1]: Kelloway, E.K. and Francis, L., 2012. Longitudinal research and data analysis. In *Research methods in occupational health psychology* (pp. 374-394). Routledge.

[^2]: Ribeiro, C. and Freitas, A.A., 2019. A mini-survey of supervised machine learning approaches for coping with ageing-related longitudinal datasets. In 3rd Workshop on AI for Aging, Rehabilitation and Independent Assisted Living (ARIAL), held as part of IJCAI-2019.

[^3]: Ribeiro, C. and Freitas, A.A., 2024. A lexicographic optimisation approach to promote more recent features on longitudinal decision-tree-based classifiers: applications to the English Longitudinal Study of Ageing. *Artificial Intelligence Review*, 57(4), p.84.
