<!--suppress HtmlDeprecatedAttribute -->
<div align="center">
   <p align="center">
   <h1 align="center">
      <br>
      <a href="https://i.imgur.com/jCtPpTF.png"><img src="https://i.imgur.com/jCtPpTF.png" alt="Scikit-longitudinal" width="200"></a>
      <br>
      Scikit-longitudinal
      <br>
   </h1>
   <h4 align="center">A specialised Python library for longitudinal data analysis built on Scikit-learn</h4>
   <table align="center">
      <tr>
         <td align="center">
            <h3>⚙️ Project Status</h3>
         </td>
         <td align="center">
            <h3>☎️ Contacts</h3>
         </td>
      </tr>
      <tr>
         <td valign="top">
            <!-- Python-related badges table -->
            <table>
               <tr>
                  <table>
                     <tr>
                        <td>
                           <a href="https://pdm.fming.dev">
                           <img alt="pdm" src="https://img.shields.io/badge/pdm-managed-blue?style=for-the-badge&logo=python">
                           </a>
                        </td>
                        <td>
                           <a href="https://pytest.org/">
                           <img alt="pytest" src="https://img.shields.io/badge/pytest-passing-green?style=for-the-badge&logo=pytest">
                           </a><br />
                           <a href="https://codecov.io/gh/Scikit-Longitudinal/Scikit-Longitudinal">
                           <img alt="Codecov" src="https://img.shields.io/badge/coverage-88%25-brightgreen.svg?style=for-the-badge&logo=appveyor">
                           </a>
                        </td>
                     </tr>
                     <tr>
                        <td>
                           <a href="https://flake8.pycqa.org/en/latest/">
                           <img alt="flake8" src="https://img.shields.io/badge/flake8-checked-blue?style=for-the-badge&logo=python">
                           </a><br />
                           <a href="https://www.pylint.org/">
                           <img alt="pylint" src="https://img.shields.io/badge/pylint-checked-blue?style=for-the-badge&logo=python">
                           </a><br />
                           <a href="https://pre-commit.com/">
                           <img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-checked-blue?style=for-the-badge&logo=python">
                           </a>
                        </td>
                        <td>
                           <a href="https://github.com/PyCQA/isort">
                           <img alt="isort" src="https://img.shields.io/badge/isort-compliant-green?style=for-the-badge&logo=python">
                           </a><br />
                           <a href="https://github.com/psf/black">
                           <img alt="black" src="https://img.shields.io/badge/black-formatted-black?style=for-the-badge&logo=python">
                           </a><br />
                           <a href="https://github.com/hhatto/autopep8">
                           <img alt="autopep8" src="https://img.shields.io/badge/autopep8-compliant-green?style=for-the-badge&logo=python">
                           </a>
                        </td>
                     </tr>
                  </table>
                  <td valign="center">
                     <table>
                        <tr>
                           <td>
                                <a href="mailto:s.g.provost@kent.ac.uk">
                                    <img alt="Microsoft Outlook" src="https://upload.wikimedia.org/wikipedia/commons/d/df/Microsoft_Office_Outlook_%282018%E2%80%93present%29.svg" width="40" height="40">
                                </a><br />
                                <a href="https://linkedin.com/in/simonprovostdev/">
                                    <img alt="LinkedIn" src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="40" height="40">
                                </a><br />
                                <a href="https://stackoverflow.com/users/9814037/simon-provost">
                                    <img alt="Stack Overflow" src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" width="40" height="40">
                                </a><br />
                                <a href="https://scholar.google.com/citations?user=Lv_LddYAAAAJ">
                                    <img alt="Google Scholar" src="https://upload.wikimedia.org/wikipedia/commons/c/c7/Google_Scholar_logo.svg" width="40" height="40">
                                </a>
                            </td>
                        </tr>
                     </table>
                  </td>
               </tr>
            </table>
         </td>
      </tr>
   </table>
</div>

> 🌟 **Exciting Update**: We're delighted to introduce the brand new v0.1 documentation for Scikit-longitudinal! For a
> deep dive into the library's capabilities and features,
> please [visit here](https://simonprovost.github.io/scikit-longitudinal/).

## <a id="about-the-project"></a>💡 About The Project

`Scikit-longitudinal` (Sklong) is a machine learning library designed to analyse
longitudinal data (Classification tasks focussed as of today). It offers tools and models for processing, analysing,
and predicting longitudinal data, with a user-friendly interface that
integrates with the `Scikit-learn` ecosystem.

Please for further information, visit the [official documentation](https://simonprovost.github.io/scikit-longitudinal/).

## <a id="installation"></a>🛠️ Installation

To install `Sklong`, take these two easy steps:

1. ✅ **Install the latest version of `Sklong`**:

```shell
pip install Scikit-longitudinal
```
You could also install different versions of the library by specifying the version number, 
e.g. `pip install Scikit-longitudinal==0.0.1`. 
Refer to [Release Notes](https://github.com/simonprovost/scikit-longitudinal/releases)

2. 📦 **[MANDATORY] Update the required dependencies (Why? See [here](https://github.com/pdm-project/pdm/issues/1316#issuecomment-2106457708))**

`Scikit-longitudinal` incorporates a modified version of `Scikit-Learn` called `Scikit-Lexicographical-Trees`, 
which can be found at [this Pypi link](https://pypi.org/project/scikit-lexicographical-trees/).

This revised version guarantees compatibility with the unique features of `Scikit-longitudinal`. 
Nevertheless, conflicts may occur with other dependencies in `Scikit-longitudinal` that also require `Scikit-Learn`. 
Follow these steps to prevent any issues when running your project.

<details>
<summary><strong>🫵 Simple Setup: Command Line Installation</strong></summary>

Say you want to try `Sklong` in a very simple environment. Such as without a proper `project.toml` file (`Poetry`, `PDM`, etc).
Run the following command:

```shell
pip uninstall scikit-learn && pip install scikit-lexico-trees
```

*Note: Although the main installation command install both, yet it’s advisable to verify the correct versions used is 
`Scikit-Lexicographical-trees` to prevent conflicts.*
</details>

<details>
<summary><strong>🫵 Project Setup: Using `PDM` (or any other such as `Poetry`, etc.)</strong></summary>

Imagine you have a project being managed by `PDM`, or any other package manager. The example below demonstrates `PDM`. 
Nevertheless, the process is similar for `Poetry` and others. Consult their documentation for instructions on excluding a 
package.

Therefore, to prevent dependency conflicts, you can exclude `Scikit-Learn` by adding the provided configuration 
to your `pyproject.toml` file.

```toml
[tool.pdm.resolution]
excludes = ["scikit-learn"]
```

*This exclusion ensures Scikit-Lexicographical-Trees (used as `Scikit-learn`) is used seamlessly within your project.*
</details>

### 💻 Developer Notes

For developers looking to contribute, please refer to the `Contributing` section of the [official documentation](https://simonprovost.github.io/scikit-longitudinal/).

## <a id="Supported-Operating-Systems"></a>🛠️ Supported Operating Systems

`Scikit-longitudinal` is compatible with the following operating systems:

- MacOS  
- Linux 🐧
- Windows via Docker only (Docker uses Linux containers) 🪟 (To try without but we haven't tested it)

## <a id="how-to-use"></a></a>🚀 Getting Started

To perform longitudinal analysis with `Scikit-Longitudinal`, use the
`LongitudinalDataset` class to prepare the dataset. To analyse your
data, use the `LexicoGradientBoostingClassifier` _(i.e. Gradient Boosting variant for Longitudinal Data)_ or another
available
estimator/preprocessor.

Following that, you can apply the popular _fit_, _predict_, _prodict_proba_, or _transform_
methods in the same way that `Scikit-learn` does, as shown in the example below.

``` py
from scikit_longitudinal.data_preparation import LongitudinalDataset
from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_gradient_boosting import LexicoGradientBoostingClassifier

dataset = LongitudinalDataset('./stroke_4_years.csv')
dataset.load_data_target_train_test_split(
  target_column="class_stroke_wave_4",
)

# Pre-set or manually set your temporal dependencies 
dataset.setup_features_group(input_data="Elsa")

model = LexicoGradientBoostingClassifier(
  features_group=dataset.feature_groups(),
  threshold_gain=0.00015
)

model.fit(dataset.X_train, dataset.y_train)
y_pred = model.predict(dataset.X_test)
```

## <a id="citation"></a>📝 How to Cite?

Paper has been submitted to a conference. In the meantime, for the repository, utilise the button top right corner of the
repository "How to cite?", or open the following citation file: [CITATION.cff](./CITATION.cff).

## <a id="license"></a>🔐 License

[MIT License](./LICENSE)