<!--suppress HtmlDeprecatedAttribute -->
<div align="center">
   <p align="center">
   <h1 align="center">
      <br>
      <a href="https://i.imgur.com/jCtPpTF.png">
         <img src="https://i.imgur.com/jCtPpTF.png" alt="Scikit-longitudinal" width="200">
      </a>
      <br>
      Scikit-longitudinal
      <br>
   </h1>
   <h4 align="center">A specialised Python library for longitudinal data analysis built on Scikit-learn</h4>
</div>

<div align="center">

<!-- All badges in a row -->
<a href="https://github.com/openml-labs/gama">
   <img src="https://img.shields.io/badge/Fork-SKLEARN-green?labelColor=Purple&style=for-the-badge"
        alt="Fork Sklearn" />
</a>
<a href="https://pytest.org/">
   <img alt="pytest" src="https://img.shields.io/badge/pytest-passing-green?style=for-the-badge&logo=pytest">
</a>
<a href="https://codecov.io/gh/Scikit-Longitudinal/Scikit-Longitudinal">
   <img alt="Codecov" src="https://img.shields.io/badge/coverage-88%25-brightgreen.svg?style=for-the-badge&logo=appveyor">
</a>
<a href="https://www.pylint.org/">
   <img alt="pylint" src="https://img.shields.io/badge/pylint-checked-blue?style=for-the-badge&logo=python">
</a>
<a href="https://pre-commit.com/">
   <img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-checked-blue?style=for-the-badge&logo=python">
</a>
<a href="https://github.com/psf/black">
   <img alt="black" src="https://img.shields.io/badge/black-formatted-black?style=for-the-badge&logo=python">
</a>
<a href="https://github.com/astral-sh/ruff">
   <img alt="Ruff" src="https://img.shields.io/badge/Linter-Ruff-brightgreen?style=for-the-badge">
</a>
<a href="https://github.com/astral-sh/uv">
   <img alt="UV Managed" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json">
</a>

[simonprovostdev.vercel.app](https://simonprovostdev.vercel.app/)

</div>

---

# 📰 Latest News

- **Updated Workflow**: Now leveraging [UV](https://docs.astral.sh/uv/) for enhanced project management and dependency resolution.
- **Documentation**: Dive into Scikit-longitudinal's features and capabilities in our [official documentation](https://simonprovost.github.io/scikit-longitudinal/).
- **PyPI Availability**: The library is available on [PyPI](https://pypi.org/project/Scikit-longitudinal/).

---

## <a id="about-the-project"></a>💡 About The Project

`Scikit-longitudinal` (Sklong) is a machine learning library designed to analyse
longitudinal data (Classification tasks focussed as of today). It offers tools and models for processing, analysing,
and predicting longitudinal data, with a user-friendly interface that
integrates with the `Scikit-learn` ecosystem.

For more details, visit the [official documentation](https://simonprovost.github.io/scikit-longitudinal/).

---

## <a id="installation"></a>🛠️ Installation

To install Scikit-longitudinal:

1. ✅ Install the latest version:
   ```bash
   pip install Scikit-longitudinal
   ```

   To install a specific version:
   ```bash
   pip install Scikit-longitudinal==0.1.0
   ```

See further in the [Quick Start of the documentation](https://simonprovost.github.io/scikit-longitudinal/quick-start) for more details.

---

## <a id="getting-started"></a>🚀 Getting Started

Here's how to analyse longitudinal data with Scikit-longitudinal:

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

See further in the [Quick Start of the documentation](https://simonprovost.github.io/scikit-longitudinal/quick-start) for more details.

---

## <a id="citation"></a>📝 How to Cite

If you find Scikit-longitudinal helpful, please cite us using the `CITATION.cff` file or via the "Cite this repository" button on GitHub.

---

## <a id="license"></a>🔐 License

Scikit-longitudinal is licensed under the [MIT License](./LICENSE).
