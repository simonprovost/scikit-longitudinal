---
hide:
 - navigation
---

# Contributing to `Sklong`

!!! important "Project Status"
    Scikit-longitudinal is actively evolving. Expect changes, and if you encounter issues, please open a [GitHub Issue](https://github.com/simonprovost/scikit-longitudinal/issues).

!!! tip "New Contributors"
    Explore [GitHub Issues](https://github.com/simonprovost/scikit-longitudinal/issues) for beginner-friendly tasks or reach out for guidance.

---

## Getting Started

!!! warning "Windows support"
    Native `Windows` installation is not our recommended path at the moment.
    The dependency stack is maintained and validated on `macOS` and stable Linux environments first, so the most reliable setup on Windows is to use `Google Colab` or a Linux-based `Docker` image.
    If you report an installation issue, include your Python version, whether you are using native Windows or Docker, and the full command output.

### Prerequisites
- **Python 3.10–3.13**: [Download](https://www.python.org/downloads/)
- **UV**: [Installation Guide](https://docs.astral.sh/uv/)

### Environment Setup

=== "Using UV <span class='tab-badge tab-badge--accent'>Recommended</span>"

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
    source .venv/bin/activate # On Windows: .venv\Scripts\activate
    ```

    4. **Install Dependencies**:

    ```bash
    uv sync --all-groups
    ```

    !!! warning "Windows users"
        Native `Windows` setup is not the recommended contributor workflow.
        For the smoothest experience, use a Linux-based `Docker` image or work in `Google Colab` for quick validation.
        If you need Docker locally, start with the [Docker setup guide](https://docs.docker.com/get-docker/).

=== "Alternative Tools"
    Prefer `pip` or `conda`? You can adapt the setup, but UV is recommended for its speed and efficiency.

---

## Linting and Formatting
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

## Pre-Commit Hooks
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

## Adding New Components
Scikit-longitudinal currently exposes shared extension templates for three component families: classifiers, transformers, and data-preparation tools. Regressors do not yet have an equivalent shared base template, so new regressor work should follow the existing lexicographical regressor implementations more closely.

=== "Estimators"

    Add new classifier primitives to `estimators/`.

    1. **Location**: Create the implementation in the most appropriate estimator package, such as `estimators/ensemble/` or `estimators/trees/`.
    2. **Class Definition**: Inherit from `CustomClassifierMixinEstimator` from `scikit_longitudinal.templates`.
    3. **Implementation**: Implement `_fit`, `_predict`, and `_predict_proba`. The public `fit`, `predict`, and `predict_proba` methods are already provided by the template and perform input validation for you.
    4. **Temporal Metadata**: Accept and store `features_group` or any other longitudinal metadata your classifier needs.
    5. **Exports**: Update the relevant `__init__.py` when you want the class to be importable from the public package surface. Discovery scans modules automatically, so public exports and discovery are related but not the same thing.

    **Example**:

    ```python
    import numpy as np
    from overrides import override

    from scikit_longitudinal.templates import CustomClassifierMixinEstimator

    class MyClassifier(CustomClassifierMixinEstimator):
        def __init__(self, features_group=None):
            self.features_group = features_group
            self._majority_class = None

        @override
        def _fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
            _ = X, sample_weight
            values, counts = np.unique(y, return_counts=True)
            self.classes_ = values
            self._majority_class = values[np.argmax(counts)]
            return self

        @override
        def _predict(self, X: np.ndarray) -> np.ndarray:
            return np.full(X.shape[0], self._majority_class)

        @override
        def _predict_proba(self, X: np.ndarray) -> np.ndarray:
            proba = np.zeros((X.shape[0], len(self.classes_)))
            majority_index = np.where(self.classes_ == self._majority_class)[0][0]
            proba[:, majority_index] = 1.0
            return proba
    ```

=== "Preprocessors"

    Add data transformation tools to `preprocessors/`.

    1. **Location**: Create the implementation in the relevant preprocessing package, such as `preprocessors/feature_selection/`.
    2. **Class Definition**: Inherit from `CustomTransformerMixinEstimator`.
    3. **Implementation**: Implement `_fit` and `_transform`. The public `fit` and `transform` methods are already provided by the template and validate inputs before delegation.
    4. **Exports**: Update the relevant `__init__.py` when you want the preprocessor exposed as a stable public import.

    **Example**:

    ```python
    import numpy as np
    from overrides import override

    from scikit_longitudinal.templates import CustomTransformerMixinEstimator

    class MyPreprocessor(CustomTransformerMixinEstimator):
        def __init__(self, keep_first_n: int = 5):
            self.keep_first_n = keep_first_n

        @override
        def _fit(self, X: np.ndarray, y: np.ndarray = None):
            _ = y
            self.selected_indices_ = list(range(min(self.keep_first_n, X.shape[1])))
            return self

        @override
        def _transform(self, X: np.ndarray) -> np.ndarray:
            return X[:, self.selected_indices_]
    ```

=== "Data Preparation Tools"

    Add utilities to `data_preparation/`.

    1. **Location**: Create a new module in `data_preparation/`, for example `my_data_tool.py`.
    2. **Class Definition**: Inherit from `DataPreparationMixin`.
    3. **Required Method**: Implement `_prepare_data`. That is the only method required by the mixin itself.
    4. **Optional Transformation Stage**: Add `_transform()` when your tool follows the same pattern as existing preparation components that first cache input state through `prepare_data(...)` and then expose a transformation step for downstream pipeline helpers.
    5. **Exports**: Update `__init__.py` when you want a stable public import path.

    **Example**:

    ```python
    import numpy as np
    import pandas as pd
    from overrides import override

    from scikit_longitudinal.templates import DataPreparationMixin

    class MyDataTool(DataPreparationMixin):
        def __init__(self, feature_list_names=None):
            self.feature_list_names = feature_list_names
            self.dataset_ = None
            self.target_ = None

        @override
        def _prepare_data(self, X: np.ndarray, y: np.ndarray = None):
            self.dataset_ = pd.DataFrame(X, columns=self.feature_list_names)
            self.target_ = y
            return self

        def _transform(self):
            transformed = self.dataset_.copy()
            feature_list_names = transformed.columns.tolist()
            return transformed, None, None, feature_list_names
    ```

!!! tip "Template Usage"
    Mirror the style of a nearby component, add focused tests under `scikit_longitudinal/tests/`, and update package exports when you want the new primitive to be part of the public import surface.

---

## Pipeline Architecture
Understand how components integrate:

1. **Data Loading**: Use `LongitudinalDataset` to load and prepare data.
2. **Feature Grouping**: Define `features_group` for temporal dependencies.
3. **Preprocessing**: Apply preprocessors if needed.
4. **Estimation**: Train an estimator for prediction.

---

## Generating Documentation
Update and preview docs locally:

1. **Build Docs**:

 ```bash
 ./build_docs.sh
 ```
2. **Serve Docs**:

 ```bash
 uv run zensical serve
 ```
3. **View**: Open `http://127.0.0.1:8000`.

---

## Submitting Contributions

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
    Use meaningful messages (e.g., `feat: add new estimator`) for clarity.

---

## Running Tests
Validate your changes:
```bash
uv run pytest -sv tests/
```
