
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
                           <a href="https://python-poetry.org/">
                           <img alt="poetry" src="https://img.shields.io/badge/poetry-managed-blue?style=for-the-badge&logo=python">
                           </a>
                        </td>
                        <td>
                           <a href="https://pytest.org/">
                           <img alt="pytest" src="https://img.shields.io/badge/pytest-passing-green?style=for-the-badge&logo=pytest">
                           </a><br />
                           <a href="https://codecov.io/gh/Scikit-Longitudinal/Scikit-Longitudinal">
                           <img alt="Codecov" src="https://img.shields.io/badge/coverage-95%25-brightgreen.svg?style=for-the-badge&logo=appveyor">
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
                              <a href="mailto:sgp28@kent.ac.uk">
                              <img alt="Microsoft Outlook" src="https://img.shields.io/badge/Microsoft_Outlook-0078D4?style=for-the-badge&logo=microsoft-outlook&logoColor=white">
                              </a><br />
                              <a href="https://linkedin.com/in/simonprovostdev/">
                              <img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white">
                              </a><br />
                              <a href="https://discord.com/users/Simon__#6384">
                              <img alt="Discord" src="https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white">
                              </a><br />
                            <a href="https://stackoverflow.com/users/9814037/simon-provost">
                              <img alt="Stack Overflow" src="https://img.shields.io/badge/Stack_Overflow-FE7A16?style=for-the-badge&logo=stack-overflow&logoColor=white">
                              </a><br />
                              <a href="https://twitter.com/SimonProvost_">
                              <img alt="Twitter" src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white">
                              </a><br />
                              <a href="https://scholar.google.com/citations?user=Lv_LddYAAAAJ&hl=en&authuser=3">
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
<!-- TABLE OF CONTENTS -->
<details open="open">
   <summary style="font-size: 1.5em; font-weight: bold;">
      📚 Table of Contents
   </summary>
   <ol>
      <li><a href="#about-the-project">💡 About The Project</a></li>
      <li><a href="#key-features">⭐️ Key Features</a></li>
      <li><a href="#installation">🛠️ Installation (ON-HOLD)</a></li>
      <li><a href="#how-to-use">🚀 Getting Started (ON-HOLD)</a></li>
      <li><a href="#documentation">📖 Documentation (ON-HOLD)</a></li>
      <li>
         <a href="#contributing">🤝 Contributing (developers)</a>
         <ul>
            <li><a href="#setup">Setup</a></li>
            <li><a href="#code-convention">Coding Conventions</a></li>
            <li><a href="#pull-request-process">Pull Request Process</a></li>
         </ul>
      </li>
      <li><a href="#faq">❓ FAQ</a></li>
      <li><a href="#citation">📝 How to Cite?</a></li>
      <li><a href="#related">🔗 Related</a></li>
      <li><a href="#license">🔐 License</a></li>
      <li><a href="#contact">📞 Contact</a></li>
   </ol>
</details>


## 💡 About The Project

Scikit-longitudinal is a machine learning library specifically designed for longitudinal data analysis. It provides a collection of tools and models to process, analyze, and make predictions on longitudinal data, with a simple and user-friendly interface compatible with the Scikit-learn ecosystem.

## ⭐️Key Features

### 📈 Classifier estimators

* Nested Tree Classifier (main code is available [here](scikit_longitudinal/estimators/tree/nested_tree/nested_tree.py))
* Lexicographical Random Forest (main code is available [here](scikit_longitudinal/estimators/tree/lexico_rf/lexico_rf.py))

### 📉 Feature Selection estimators

* Correlation-based Feature Selection Per Group (CFS-PerGroup version 1 and 2) (main code is available [here](scikit_longitudinal/preprocessing/feature_selection/cfs_per_group/cfs_per_group.py))

## 🛠️ Installation (ON-HOLD until the first public release).

_TODO: Describe how to install the package, including any dependencies._

## 🚀 Getting Started (ON-HOLD until the first public release).

````python
import scikit_longitudinal as skl

# Load your data
data = ...

# Initialize the Scikit-longitudinal model
model = skl.<your_desired_estimator>()

# Train the model
model.fit(data)

# Make predictions
predictions = model.predict(new_data)
````

## 📖 Documentation (ON-HOLD until the first public release).

Use `make docs` to build the documentation locally or for detailed documentation, including tutorials and API reference,
please visit our [official documentation](https://simonprovost.github.io/scikit-longitudinal/).

## 🤝 Contributing (developers)


### Setup

> ⚠️ **DISCLAIMER**: This project is still under development, and the setup is not yet fully automated. Furthermore, it has been tested only on macOS for now. It should work on Linux distributions, but we are not sure about Windows.

To set up the development environment, please follow these steps:

<details>
  <summary>📌 Prerequisites</summary>

  * Ensure that [Poetry](https://python-poetry.org/docs/#installation) and [Pipenv](https://pipenv.pypa.io/en/latest/install/#installing-pipenv) are installed.
  * Ensure that [Make](https://www.gnu.org/software/make/) and [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) are installed.
  * Ensure that [LibOmp](https://www.openmp.org/resources/openmp-compilers-tools/) is installed ([Recommended for macOS](https://formulae.brew.sh/formula/libomp)).
  * Export necessary environment variables in your shell configuration file (e.g., `.bashrc`, `.zshrc`, or `config.fish` if you are using the fish shell). Open an issue if you need help at this stage.
  * [macOS] Ensure that [Xcode](https://developer.apple.com/xcode/) is installed.
  * [macOS] Ensure that [Homebrew](https://brew.sh/) is installed.
  * [macOS] Ensure that `SDKROOT` is exported. It is usually available at `/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/`.

</details>

1. Clone the repository: `git clone git@github.com:simonprovost/scikit-longitudinal.git`
2. Create a `.env` file in the root directory of the project and add the following environment variables available in the `.env.example` file.
3. Use the Makefile target rule `install_dev` to install the development dependencies:
    ```
    make install_dev
    ```
    > 📝 This command will install the development dependencies, create a Poetry virtual environment, install the package in editable mode, and run the tests. If this fails, please open an issue.

🎉 Voilà! You are ready to contribute!


### Coding Conventions
We follow the [Karma Git Commit Convention](http://karma-runner.github.io/6.4/dev/git-commit-msg.html) for commit
messages and a modified version of the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
with fewer restrictions for naming conventions and coding style. Please familiarize yourself with these conventions
before contributing.

#### File and Class Function Naming Conventions

- **File names:** Snake_case naming convention is used for file names. Snake_case uses lowercase words separated by underscores (e.g., `file_name.py`).
- **Class names**: The PascalCase (or UpperCamelCase) convention is employed (e.g., `ClassName`).
- **Function and method names:** The snake_case naming convention is also used for function and method names. These names should be lowercase with words separated by underscores (e.g., `function_name()` or `method_name()`).

### Pull Request Process

To submit a pull request, please follow these steps:

* Fork the repository: Click the "Fork" button at the top right corner of the repository page to create a copy of the project in your GitHub account.
* Create a new branch: In your forked repository, create a new branch named after the feature or fix you are working on (e.g., feature/new-feature-name or fix/bug-fix-name).
* Make your changes: Implement the desired feature or fix in your new branch, and make sure to adhere to the project's coding conventions.
* Commit your changes: Use clear and concise commit messages, following the Karma Git Commit Convention. Make sure to include any necessary tests, documentation updates, or code style adjustments.
* Submit a pull request: Click the "New Pull Request" button in your forked repository and select your newly created branch as the source. Then, target the main branch of the original repository as the destination. Provide a detailed description of the changes you've made, and submit the pull request.

* Once your pull request is submitted, maintainers will review your changes and provide feedback. Be prepared to make any necessary adjustments, and collaborate with the maintainers to get your contribution merged.

## ❓ FAQ

Explore the properties of Longitudinal Data and Time-Series Data in this comprehensive FAQ.

**Q: What is Longitudinal Data?**

A: Longitudinal Data refers to observations made on multiple variables of interest for the same subject over an extended period of time. This type of data is particularly valuable for studying changes and trends, as well as making predictions about future outcomes. For example, in a medical study, a patient's health measurements such as blood pressure, heart rate, and weight might be recorded over several years to analyze the effectiveness of a treatment.

**Q: What are the differences between Time-Series Data and Longitudinal Data?**

A: Time-Series Data and Longitudinal Data both involve observations made over time, but they differ in several aspects:

Focus: Time-Series Data focuses on a single variable measured at regular intervals, while Longitudinal Data involves multiple variables observed over time for each subject.
Nature: Time-Series Data is usually employed for continuous data, whereas Longitudinal Data can handle both continuous and categorical data.
Time gap: Time-Series Data typically deals with shorter time periods (e.g., seconds, minutes, or hours), while Longitudinal Data often spans longer durations (e.g., months or years).
In summary, the main differences between Time-Series and Longitudinal Data lie in the focus, nature, and the length of the time intervals considered.

## 📝 How to Cite?

If you use Scikit-Longitudinal in your research, please cite our paper and our repoitiory:

For the repository, utilise the button top right corner of the repository "How to cite?",

For the paper, use the following BibTeX entry:

_TODO: Add citation information for the Scikit-Longitudinal papers._

## 🔗 Related

- Auto-prognosis: [GitHub](https://github.com/vanderschaarlab/autoprognosis)
- Clairvoyance: [GitHub](https://github.com/vanderschaarlab/clairvoyance)

## 🔐 License

MIT License
