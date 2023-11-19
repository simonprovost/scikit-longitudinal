
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
                           <img alt="Codecov" src="https://img.shields.io/badge/coverage-85%25-brightgreen.svg?style=for-the-badge&logo=appveyor">
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
   </ol>
</details>

> 🌟 **Exciting Update**: We're delighted to introduce the brand new v0.1 documentation for Scikit-longitudinal! For a deep dive into the library's capabilities and features, please [visit here](https://simonprovost.github.io/scikit-longitudinal/).

## <a id="about-the-project"></a>💡 About The Project

Scikit-longitudinal is a machine learning library designed specifically for the analysis
of longitudinal data. It provides a collection of tools and models for processing,
analysing, and making predictions on longitudinal data, with an easy-to-use interface
that is compatible with the Scikit-learn ecosystem.

For Neural Networks based models, we
recommend to look in the related projecs available in the [Related](#related) section - therefore,
this current project will not provide any Neural Networks based models as of today.

## <a id="key-features"></a>⭐️Key Features

We recommend you to open the [scikit_longitudinal folder's readme](scikit_longitudinal/README.md) file to see the table of key features.

## <a id="installation"></a>🛠️ Installation

**ON-HOLD until the first public release**

_TODO: Describe how to install the package, including any dependencies._

## <a id="how-to-use"></a></a>🚀 Getting Started

ON-HOLD until the first public release

_TODO: Describe how to easily use the package with a code snippet._

## <a id="documentation"></a>📖 Documentation

ON-HOLD until the first public release

_TODO: Describe how to access the documentation. Try Sphinx and Pdoc3._

## <a id="contributing"></a>🤝 Contributing (developers)

> ⚠️ **DISCLAIMER**: This project is still under development, and the setup is not yet fully automated. It has been tested on macOS and Linux distributions. The assurance of Windows compatibility is currently not guaranteed. However, our Project Packages and Dependencies Manager (PDM) enables cross-compatibility.

### 📌 Prerequisites

#### Common Requirements
- [Python 3.9.8](https://www.python.org/downloads/release/python-398/). For managing multiple Python versions, [Pyenv](https://github.com/pyenv/pyenv) is recommended.
- [PDM (Python Dependency Management)](https://pdm.fming.dev)
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

#### Linux-specific Requirements
- `libomp-dev` from `apt-get`:
  ```bash
  sudo apt-get install libomp-dev
  ```

#### macOS-specific Requirements
- [Xcode](https://developer.apple.com/xcode/) - Make sure to open XCODE and accept the license agreement.
- [Homebrew](https://brew.sh/)
- `SDKROOT` environment variable, typically located at `/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/`. Or what also works is to run:
  ```bash
  export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
  ```
- `libomp` from `brew`:
  ```bash
  brew install libomp
  ```

### <a id="setup"></a>🛠 Manual Setup Instructions for macOS or Linux Environments

To manually configure your environment, please adhere to the following procedure meticulously:

1. **Setting up the package manager:**
    - Initialize the package manager with Conda as the backend for virtual environments:
      ```bash
      pdm config venv.backend conda
      ```

2. **Selecting Python version:**
    - Specify the Python version for the project. Here, we are selecting Python 3.9:
      ```bash
      pdm use 3.9
      ```

3. **Project Setup:**
    - Execute the setup script. This step may prompt you to export certain variables or configure compilers like GCC or Clang. Please comply with the on-screen instructions:
      ```bash
      pdm run setup_project
      ```

4. **Environment Variables Configuration:**
    - Set the `PDM_IN_ENV` variable to `in-project` to ensure that the package manager operates within the project directory:
      ```bash
      export PDM_IN_ENV=in-project
      ```

5. **Conda Initialization:**
    - Initialize Conda for your shell. Replace `bash` with `zsh` or `fish` as per your shell preference:
      ```bash
      conda init bash 
      ```

6. **Shell Configuration:**
    - Source your shell configuration file to apply the changes. Again, replace `.bashrc` with the appropriate file name corresponding to your shell:
      ```bash
      source ~/.bashrc # Replace with ~/.zshrc or ~/.config/fish/config.fish accordingly
      ```

7. **Activating Virtual Environment:**
    - Activate the virtual environment with the following command:
      ```bash
      eval $(pdm venv activate $PDM_IN_ENV)
      ```

8. **Project Dependencies Installation:**
    - Install all the project dependencies by running:
      ```bash
      pdm run install_project
      ```

#### Troubleshooting Errors

If you encounter any errors during the setup process and are unsure how to resolve them, please follow these troubleshooting steps:

1. **Deactivate Conda Environment**:
   ```bash
   conda deactivate
   ```
   
2. **Clear PDM Cache**:
   ```bash
   pdm cache clear
   ```
   
3. **Remove Pypackages Directory (subject of many errors from time to time)**:
   ```bash
   rm -rf __pypackages__/
   ```
   
4. **Remove PDM Virtual Environment**:
   ```bash
   pdm venv remove_env
   ```
   
After following these steps, try to reinstall the project dependencies. If the issue persists, 
feel free to open an issue on the GitHub repository for additional support.

### WINDOWS KNOWN ISSUES

- Git Handling Lines Ending: We recommend that you setup the following git configuration to avoid Windows to automatically add `\r\n` ending symbol line that Linux/MacOS do not understand. To configure, use [this command](https://docs.github.com/en/get-started/getting-started-with-git/configuring-git-to-handle-line-endings#global-settings-for-line-endings):
  ```bash
  git config --global core.autocrlf true
  ```

### 🐳 Docker Setup (Linux, Python 3.9.8)

#### Prerequisites
- [Docker](https://docs.docker.com/get-docker/)

#### Recommended Steps using [JetBrains (PyCharm) Docker Services](https://www.jetbrains.com/help/pycharm/docker.html)
1. Build the Docker image using JetBrains Docker Services. Scroll at the top of the Dockerfile and click on the green arrow to build the image.
2. Run the container in interactive mode using JetBrains Docker Services. On the Docker build window that might have appeared. Click on the "terminal" tab that shows up and you should be able to run the command interpreter inside the container.

#### Manual Steps (if not using JetBrains)
1. Build the Docker image manually. Follow the [Docker documentation](https://docs.docker.com/) to build the image.
2. Run the container in interactive mode manually.

#### Common Steps
1. Inside the Docker container, activate your PDM-based Conda environment by running:
   ```bash
   eval $(pdm venv activate $PDM_IN_ENV)
   # Alternatively, you can run:
   # pdm venv use $PDM_IN_ENV
   # conda activate the returned path
   ```
2. You can now execute your scripts or modify the Dockerfile to include them.
3. For testing purposes, run _(Note: If you do not have the entire ELSA Databases, you can contact us, or ignore the failed tests because of missing data)_
   ```bash
   pdm run tests
   ```

🎉 Voilà! You are ready to contribute!

### <a id="code-convention"></a>✒️ Coding Conventions
We follow the [Karma Git Commit Convention](http://karma-runner.github.io/6.4/dev/git-commit-msg.html) for commit
messages and a modified version of the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
with fewer restrictions for naming conventions and coding style. Please familiarize yourself with these conventions
before contributing.

#### File and Class Function Naming Conventions:

- **File names:** Snake_case naming convention is used for file names. Snake_case uses lowercase words separated by underscores (e.g., `file_name.py`).
- **Class names**: The PascalCase (or UpperCamelCase) convention is employed (e.g., `ClassName`).
- **Function and method names:** The snake_case naming convention is also used for function and method names. These names should be lowercase with words separated by underscores (e.g., `function_name()` or `method_name()`).

### <a id="pull-request-process"></a>📥 Pull Request Process

To submit a pull request, please follow these steps:

* Fork the repository: Click the "Fork" button at the top right corner of the repository page to create a copy of the project in your GitHub account.
* Create a new branch: In your forked repository, create a new branch named after the feature or fix you are working on (e.g., feature/new-feature-name or fix/bug-fix-name).
* Make your changes: Implement the desired feature or fix in your new branch, and make sure to adhere to the project's coding conventions.
* Commit your changes: Use clear and concise commit messages, following the Karma Git Commit Convention. Make sure to include any necessary tests, documentation updates, or code style adjustments.
* Submit a pull request: Click the "New Pull Request" button in your forked repository and select your newly created branch as the source. Then, target the main branch of the original repository as the destination. Provide a detailed description of the changes you've made, and submit the pull request.

* Once your pull request is submitted, maintainers will review your changes and provide feedback. Be prepared to make any necessary adjustments, and collaborate with the maintainers to get your contribution merged.

## <a id="faq"></a>❓ FAQ

Explore the properties of Longitudinal Data and Time-Series Data in this comprehensive FAQ.

**Q: What is Longitudinal Data?**

A: Longitudinal Data refers to observations made on multiple variables of interest for the same subject over an extended period of time. This type of data is particularly valuable for studying changes and trends, as well as making predictions about future outcomes. For example, in a medical study, a patient's health measurements such as blood pressure, heart rate, and weight might be recorded over several years to analyze the effectiveness of a treatment.

**Q: What are the differences between Time-Series Data and Longitudinal Data?**

A: Time-Series Data and Longitudinal Data both involve observations made over time, but they differ in several aspects:

* **Focus**: _Originally_ Time-Series Data focuses on a single variable measured at regular intervals, while Longitudinal Data involves multiple variables observed over time for each subject.
* **Nature**: Time-Series Data is usually employed for continuous data, whereas Longitudinal Data can handle both continuous and categorical data.
* **Time gap**: Time-Series Data typically deals with shorter time periods (e.g., seconds, minutes, or hours), while Longitudinal Data often spans longer durations (e.g., months or years).
* **Machine Learning**: Time-Series Data are frequently used to predict future values, whereas Longitudinal Data are more frequently used to predict future outcomes. In addition, the ML algorithms used for time-series are frequently distinct from those used for longitudinal data. For instance, Time-Series based techniques are based on time-windowing techniques, whereas Longitudinal based techniques frequently use the current standard for machine learning classification for the prediction task. Nevertheless, they will adapt (create variant of these standard classification based machine learning algorithm) to comprehend the temporal nature of the data.


In summary, the main differences between Time-Series and Longitudinal Data lie in the focus, nature, and the length of the time intervals considered.

## <a id="citation"></a>📝 How to Cite?

If you use Scikit-Longitudinal in your research, please cite our paper and our repoitiory:

For the repository, utilise the button top right corner of the repository "How to cite?",

For the paper, use the following BibTeX entry:

_TODO: Add citation information for the paper we should publish about this library._

## <a id="related"></a>🔗 Related

- Auto-prognosis: [GitHub](https://github.com/vanderschaarlab/autoprognosis)
- Clairvoyance: [GitHub](https://github.com/vanderschaarlab/clairvoyance)

## <a id="license"></a>🔐 License

[MIT License](./LICENSE)
