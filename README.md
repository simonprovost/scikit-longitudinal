
<!--suppress HtmlDeprecatedAttribute -->
<div align="center">
   <p align="center">
   <h1 align="center">
      <br>
      <a href="./logo.png"><img src="./logo.png" alt="Scikit-longitudinal" width="200"></a>
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
                              <img alt="Microsoft Outlook" src="https://img.shields.io/badge/Microsoft_Outlook-0078D4?style=for-the-badge&logo=microsoft-outlook&logoColor=white">
                              </a><br />
                              <a href="https://linkedin.com/in/simonprovostdev/">
                              <img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white">
                            <a href="https://stackoverflow.com/users/9814037/simon-provost">
                              <img alt="Stack Overflow" src="https://img.shields.io/badge/Stack_Overflow-FE7A16?style=for-the-badge&logo=stack-overflow&logoColor=white">
                              <a href="https://scholar.google.com/citations?user=Lv_LddYAAAAJ">
                              <img alt="Google Scholar" src="https://img.shields.io/badge/Google_Scholar-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white">
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

> 🌟 **Exciting Update**: We're delighted to introduce the brand new v0.1 documentation for Scikit-longitudinal! For a deep dive into the library's capabilities and features, please [visit here](https://simonprovost.github.io/scikit-longitudinal/).

## <a id="about-the-project"></a>💡 About The Project

`Scikit-longitudinal` is a machine learning library designed to analyse
longitudinal data (Classification tasks focussed as of today). It offers tools and models for processing, analysing, 
and predicting longitudinal data, with a user-friendly interface that 
integrates with the `Scikit-learn` ecosystem.

Please for further information, visit the [official documentation](https://simonprovost.github.io/scikit-longitudinal/).

## <a id="installation"></a>🛠️ Installation

**ON-HOLD until the first public release**

Note that for developers, you should follow up onto the `Contributing` tab
of the [official documentation](https://simonprovost.github.io/scikit-longitudinal/).

## <a id="how-to-use"></a></a>🚀 Getting Started

To perform longitudinal analysis with `Scikit-Longitudinal`, use the 
`LongitudinalDataset` class to prepare the dataset. To analyse your 
data, use the `LexicoGradientBoostingClassifier` _(i.e. Gradient Boosting variant for Longitudinal Data)_ or another available 
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

Paper's citation information will be added here once published. Currently, it has been submitted to a conference. In the meantime,
for the repository, utilise the button top right corner of the repository "How to cite?".
Or open the following citation file: [CITATION.cff](./CITATION.cff).

## <a id="license"></a>🔐 License

[MIT License](./LICENSE)