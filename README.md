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

<a href="https://pytest.org/">
   <img alt="pytest" src="https://img.shields.io/badge/pytest-passing-green?style=for-the-badge&logo=pytest">
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

<img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
<img src="https://img.shields.io/static/v1?label=RUFF&message=compliant&color=9C27B0&style=for-the-badge&logo=RUFF&logoColor=white" alt="RUFF compliant">
<img src="https://img.shields.io/static/v1?label=UV&message=compliant&color=2196F3&style=for-the-badge&logo=UV&logoColor=white" alt="UV compliant">
<a href="https://codecov.io/gh/Scikit-Longitudinal/Scikit-Longitudinal">
   <img alt="Codecov" src="https://img.shields.io/badge/coverage-88%25-brightgreen.svg?style=for-the-badge&logo=appveyor">
</a>
<a href="https://github.com/openml-labs/gama">
   <img src="https://img.shields.io/badge/Fork-SKLEARN-green?labelColor=Purple&style=for-the-badge"
        alt="Fork Sklearn" />
</a>
<img src="https://img.shields.io/static/v1?label=Python&message=3.9%2B%3C3.10&color=3776AB&style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+ < 3.10">

</div>

---

## <a id="about-the-project"></a>💡 About The Project

`Scikit-longitudinal` (Sklong) is a machine learning library designed to analyse
longitudinal data (Classification tasks focussed as of today). It offers tools and models for processing, analysing,
and predicting longitudinal data, with a user-friendly interface that
integrates with the `Scikit-learn` ecosystem.

For more details, visit the [official documentation](https://scikit-longitudinal.readthedocs.io/latest//).

---

## <a id="installation"></a>🛠️ Installation

> [!NOTE]
> Want to be using `Jupyter Notebook`, `Marimo`, `Google Colab`, or `JupyterLab`?
> Head to the `Getting Started` section of the documentation, we explain it all! 🎉

To install Scikit-longitudinal:

1. ✅ Install the latest version:
   ```bash
   pip install Scikit-longitudinal
   ```

   To install a specific version:
   ```bash
   pip install Scikit-longitudinal==0.1.0
   ```

> [!CAUTION]
> `Scikit-longitudinal` is currently compatible with Python versions `3.9` only. 
> Ensure you have one of these versions installed before proceeding with the installation. 
> 
> Now, while we understand that this is a limitation, we are tied for the time being because of `Deep Forest`.
> `Deep Forest` is a dependency of `Scikit-longitudinal` that is not compatible with Python versions greater than `3.9`.
> `Deep Forest` helps us with the `Deep Forest` algorithm, to which we have made some modifications to 
> welcome `Lexicographical Deep Forest`. 
> 
> To follow up on this discussion, please refer to [this github issue](https://github.com/LAMDA-NJU/Deep-Forest/issues/124).
> 
> If you encounter any errors, feel free to explore further the `installation` section in the `Getting Started` of the documentation.
> If it still doesn't work, please open an issue on GitHub.

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

---

## <a id="citation"></a>📝 How to Cite

We are currently cooking a JOSS submission, wait a bit for it! Meanwhile, click on `Cite This Repository` on the top right corner of this page to get a BibTeX entry.

---

## <a id="license"></a>🔐 License

Scikit-longitudinal is licensed under the [MIT License](./LICENSE).
