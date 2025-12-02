# 👩‍💻 Developers

!!! important "Project Status"
    Scikit-longitudinal is actively evolving. Expect changes, and if you encounter issues, please open a [GitHub Issue](https://github.com/simonprovost/scikit-longitudinal/issues).

!!! tip "New Contributors"
    Explore [GitHub Issues](https://github.com/simonprovost/scikit-longitudinal/issues) for beginner-friendly tasks or reach out for guidance.

---

## 🛠️ Installation & Troubleshooting First

- **Supported runtimes**: Python `3.10`–`3.13` on `Ubuntu`/stable Linux and `macOS`. On `Windows`, use `Google Colab` or a `Docker` image based on `Ubuntu`.
- **Need help?** Open a [GitHub Issue](https://github.com/simonprovost/scikit-longitudinal/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen) and describe your setup.

### Apple Silicon (x86_64 wheels)

Apple Silicon users should install and run `Scikit-longitudinal` with an **x86_64** CPython via `uv`, because some dependencies only distribute Intel wheels. The steps below rely entirely on `uv`—no Rosetta shell juggling or conda environment needed.

1. **Locate an Intel CPython build**
   Use `uv` to list all macOS x86_64 interpreters (Python 3.10–3.13 shown below):
   ```bash
   uv python list --all-versions --all-platforms \
     | grep 'macos-x86_64' \
     | egrep '3\.10|3\.11|3\.12|3\.13'
   ```

2. **Install your preferred version**
   Pick any listed `cpython-<version>-macos-x86_64-none` and install it:
   ```bash
   uv python install cpython-3.12.10-macos-x86_64-none
   ```

3. **Pin the interpreter for this project**
   Tell `uv` to use that Intel build whenever you work in this repo:
   ```bash
   uv python pin cpython-3.12.10-macos-x86_64-none
   ```

4. **Install project dependencies**
   ```bash
   uv sync
   ```

5. **Verify the install**
   ```bash
   uv run python -c "import scikit_longitudinal"
   ```

After pinning the x86_64 interpreter, `uv` will automatically use it for future commands, providing a smooth Apple Silicon experience.

---

## 🚀 Install From Source Scikit-Longitudinal

### Prerequisites
- **Python 3.10–3.13**: [Download](https://www.python.org/downloads/)
- **UV**: [Installation Guide](https://docs.astral.sh/uv/)

### Environment Setup

=== "Using UV (Recommended)"

    1. **Clone the Repository**:

       ```bash
       git clone https://github.com/simonprovost/scikit-longitudinal.git
       cd scikit-longitudinal
       ```

    2. **Install and Pin Python Version**:

        ```bash
        uv python install cpython-3.10.16 # or any other 3.10+ wheel
        uv python pin cpython-3.10.16 # or any other 3.10+ wheel
        ```

    3. **Create and Activate Virtual Environment**:

       ```bash
       uv venv
       source .venv/bin/activate  # On Windows: .venv\Scripts\activate
       ```

    4. **Install Dependencies**:

       ```bash
       uv sync --all-groups
       ```

    !!! warning "Windows Users"
        Due to dependency limitations, consider using Docker. Refer to [Docker Setup](https://docs.docker.com/get-docker/) for instructions.

=== "Alternative Tools"
    Prefer `pip` or `conda`? You can adapt the setup, but UV is recommended for its speed and efficiency.

---

### Verify Setup
Run the test suite to ensure everything is working:
```bash
uv run pytest -sv tests/
```

---

## 🧹 Linting and Formatting
We use **Ruff** to maintain code quality:
- **Check Issues**:
  ```bash
  uv run ruff check
  ```
- **Fix Formatting**:
  ```bash
  uv run ruff check --fix
  ```

!!! tip "Editor Integration"
    Integrate Ruff into your editor (e.g., VSCode) for real-time feedback.

---

## 🔒 Pre-Commit Hooks
Enforce standards with pre-commit hooks:
1. **Install**:
   ```bash
   uv run pre-commit install
   ```
2. **Run Manually** (optional):
   ```bash
   uv run pre-commit run --all-files
   ```

!!! note "Automatic Execution"
    Hooks run on `git commit`. Fix any failures to proceed.

---

## 🧩 Adding New Components
Scikit-longitudinal’s modular design makes it easy to extend. Below are guidelines for adding common component types:

=== "Estimators"

    Add new classifiers or regressors to `estimators/`.

    1. **Location**: Create a file in `estimators/ensemble/` or `estimators/trees/`, e.g., `my_estimator.py`.
    2. **Class Definition**: Inherit from `CustomClassifierMixinEstimator` (see `scikit_longitudinal/templates`) alongside scikit-learn's `BaseEstimator`/`ClassifierMixin` where appropriate.
    3. **Implementation**: Implement `_fit` and `_predict` (or the public `fit`/`predict` if you prefer) and pass `features_group` for temporal awareness.
    4. **Register**: Update `__init__.py` in the respective directory to add your primitive.

    **Example**:

    ```python
    from sklearn.base import BaseEstimator, ClassifierMixin

    class MyEstimator(CustomClassifierMixinEstimator):
        def __init__(self, features_group=None):
            self.features_group = features_group

        def _fit(self, X, y):
            # Training logic
            return self

        def _predict(self, X):
            # Prediction logic
            pass
    ```

=== "Preprocessors"

    Add data transformation tools to `preprocessors/`.

    1. **Location**: Create a file in `preprocessors/feature_selection/`, e.g., `my_preprocessor.py`.
    2. **Class Definition**: Inherit from `CustomTransformerMixinEstimator` (in `scikit_longitudinal/templates`) and, if needed, scikit-learn's `TransformerMixin`.
    3. **Implementation**: Implement `_fit` and `_transform` (or `fit`/`transform`) with attention to the temporal metadata.
    4. **Register**: Update `__init__.py`.

    **Example**:

    ```python
    from sklearn.base import BaseEstimator, TransformerMixin

    class MyPreprocessor(CustomTransformerMixinEstimator):
        def __init__(self):
            pass

        def _fit(self, X, y=None):
            # Fit logic
            return self

        def _transform(self, X):
            # Transform logic
            pass
    ```

=== "Data Preparation Tools"

    Add utilities to `data_preparation/`.

    1. **Location**: Create a file, e.g., `my_data_tool.py`.
    2. **Class Definition**: Inherit from `DataPreparationMixin`.
    3. **Implementation**: Implement `_prepare_data` and `_transform` to reshape the longitudinal table as intended.
    4. **Register**: Update `__init__.py`.

    **Example**:

    ```python
    from scikit_longitudinal.templates.custom_data_preparation_mixin import DataPreparationMixin

    class MyDataTool(DataPreparationMixin):
        def __init__(self, features_group=None):
            self.features_group = features_group

        def _prepare_data(self, X, y=None):
            # Preparation logic
            return self

        def _transform(self):
            # Transformation logic
            pass
    ```

!!! tip "Template Usage"
    Use provided examples as templates and adapt to your needs. Ensure compatibility with `LongitudinalDataset`.

---

## 🏗️ Pipeline Architecture
Understand how components integrate:

1. **Data Loading**: Use `LongitudinalDataset` to load and prepare data.
2. **Feature Grouping**: Define `features_group` for temporal dependencies.
3. **Preprocessing**: Apply preprocessors if needed.
4. **Estimation**: Train an estimator for prediction.

---

## 📝 Generating Documentation
Update and preview docs locally:

1. **Build Docs**:

   ```bash
   ./build_docs.sh
   ```
2. **Serve Docs**:

   ```bash
   uv run mkdocs serve
   ```
3. **View**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## 📬 Submitting Contributions

Follow this Git workflow:

1. **Create a Branch**:
   ```bash
   git checkout -b feat/your-feature
   ```
2. **Commit Changes**:
   ```bash
   git commit -m "feat: describe your change"
   ```
3. **Rebase**:
   ```bash
   git fetch origin
   git rebase origin/main
   ```
4. **Push and Open PR**:
   ```bash
   git push origin feat/your-feature
   ```
   - Submit a pull request against `main`.

!!! tip "Commit Messages"
    Use the [Conventional Commit-style prefixes from Karma](https://karma-runner.github.io/6.4/dev/git-commit-msg.html) (e.g., `feat: add new estimator`, `docs: clarify installation`).

---

## 🧪 Running Tests
Validate your changes:
```bash
uv run pytest -sv tests/
```
