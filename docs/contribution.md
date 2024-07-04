---
hide:
  - navigation
---

# 🤝 Contributing to Scikit-longitudinal
# 🤝 Contributing to Scikit-longitudinal

## Table of Contents
2. [📋 Contributing Guidelines](#contribution-guidelines)
3. [💻 Installation for Developers](#installation-for-developers)

# 📋 Contribution Guidelines

We welcome contributions from the community and are pleased to have them. Please follow this guide when logging issues or making code changes.
For the developer installation, please scroll down to the bottom of the page.

## 📧 Logging Issues

!!! warning "All issues should be created using the [new issue form](https://github.com/simonprovost/scikit-longitudinal/issues/new/choose). Clearly describe the issue including steps to reproduce when it is a bug. If it is a new feature request, describe the scope and purpose of the feature."

## 💡 Patch and Feature Contributions

!!! info "All contributions to this project should be made via a git pull request (PR) to this repository."

Here are the steps to contribute:

1. Fork this project on GitHub and clone your fork to your machine. See [GitHub documentation](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) for more information on forking a repository.
2. Create a new branch in your forked repository. The branch name should be descriptive and start with the issue number it resolves (if any).
3. Make your changes in the new branch. Please follow the coding and style guidelines described below and ensure the code passes all tests.
4. Commit your changes to your branch. Be sure to use a clear and descriptive commit message.
5. Push your changes to your GitHub fork.
6. Open a pull request from your fork to this repository. See [GitHub documentation](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork) for more information on creating a pull request from a fork.
7. A maintainer will review your pull request and provide feedback. Please be prepared to make any necessary adjustments.

## 🫡 Coding and Style Guidelines

!!! info "We follow the [Karma Git Commit Convention](http://karma-runner.github.io/6.4/dev/git-commit-msg.html) for commit messages and a modified version of the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with fewer restrictions for naming conventions and coding style. Please familiarize yourself with these conventions before contributing."

### File and Class Function Naming Conventions:

- **File names:** Snake_case naming convention is used for file names. Snake_case uses lowercase words separated by underscores (e.g., `file_name.py`).
- **Class names**: The PascalCase (or UpperCamelCase) convention is employed (e.g., `ClassName`).
- **Function and method names:** The snake_case naming convention is also used for function and method names. These names should be lowercase with words separated by underscores (e.g., `function_name()` or `method_name()`).

## 🪜 Pull Request Process

To submit a pull request, please follow these steps:
   
   * Fork the repository: Click the "Fork" button at the top right corner of the repository page to create a copy of the project in your GitHub account.
   * Create a new branch: In your forked repository, create a new branch named after the feature or fix you are working on (e.g., feature/new-feature-name or fix/bug-fix-name).
   * Make your changes: Implement the desired feature or fix in your new branch, and make sure to adhere to the project's coding conventions.
   * Commit your changes: Use clear and concise commit messages, following the Karma Git Commit Convention. Make sure to include any necessary tests, documentation updates, or code style adjustments.
   * Submit a pull request: Click the "New Pull Request" button in your forked repository and select your newly created branch as the source. Then, target the main branch of the original repository as the destination. Provide a detailed description of the changes you've made, and submit the pull request.
   
   Once your pull request is submitted, maintainers will review your changes and provide feedback. Be prepared to make any necessary adjustments, and collaborate with the maintainers to get your contribution merged.

## 💻 Installation for Developers

Please follow the instructions below for setting up your development environment.

!!! tip  "Prerequisites"
      - [Python 3.9.8](https://www.python.org/downloads/release/python-398/). For managing multiple Python versions, [Pyenv](https://github.com/pyenv/pyenv) is recommended.
      - [PDM (Python Dependency Management)](https://pdm.fming.dev)
      - [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

### Step by Step Installation Guide

!!! warning "Fully-working environment setup is not guaranteed on Windows. We recommend using a Unix-based system for
development. Such as MacOS or Linux. On Windows, Docker is recommended having been tested on Windows 10 & 11."

Prior-all, you need to open the `.env` file at the root and set:
```bash
SKLONG_PYTHON_VERSION=<your_python_version> # e.g. 3.9.8
SKLONG_PYTHON_PATH=<your_python_path> # e.g. /usr/bin/python3.9
```

Next, to manually configure your environment, please adhere to the following procedure meticulously:

1. **Setting up the package manager:**
    - Initialise the package manager with Conda as the backend for virtual environments:
      ```bash
      pdm config venv.backend conda
      ```

2. **Selecting Python version:**
    - Specify the Python version for the project. Here, we are selecting Python 3.9:
      ```bash
      pdm use 3.9
      ```
      *Here if you have more than one python version installed, you can select the desired version.*

3. **Project Setup:**
    - Execute the setup script.
      ```bash
      pdm run setup_sklong
      ```

4. **Environment Variables Configuration:**
    - Set the `PDM_IN_ENV` variable to `in-project` to ensure that the package manager operates within the project directory:
      ```bash
      export PDM_IN_ENV=in-project
      ```

5. **Conda Initialization:**
    - Initialise Conda for your shell. Replace `bash` with `zsh` or `fish` as per your shell preference:
      ```bash
      conda init bash
      ```
      *It can appears capricious, if it does not work, you can try to go ahead anyway.*

6. **Shell Configuration:**
    - Source your shell configuration file to apply the changes. Again, replace `.bashrc` with the appropriate file name corresponding to your shell:
      ```bash
      source ~/.bashrc # Replace with ~/.zshrc or ~/.config/fish/config.fish accordingly
      ```
      *It can appears capricious, if it does not work, you can try to go ahead anyway.*

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
    !!! note "Install Prod only"
          If you want to install only the production dependencies, you can run:
          ```bash
          pdm run install_prod
          ```
    !!! note "Install Dev only"
          If you want to install only the development dependencies, you can run:
          ```bash
          pdm run install_dev
          ```
   *See further information in the `pyproject.toml` file.*

9. **Running Tests:**
    - To run the tests, execute the following command:
      ```bash
      pdm run tests
      ```
    !!! warning "You need to have the dev dependencies installed to run the tests."
          If you want to install only the development dependencies, you can run:
          ```bash
          pdm run install_dev
          ```

### ❌ Troubleshooting Errors

#### General Installation Related

If you encounter any errors during the setup process and are unsure how to resolve them, please follow these troubleshooting steps

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
   pdm run remove_env
   ``` 

After following these steps, try to reinstall the project dependencies. If the issue persists, 
feel free to open an issue on the GitHub repository for additional support.

### 🐳 Docker Setup (Linux, Python 3.9.8)

!!! tip "Prerequisites"
    - [Docker](https://docs.docker.com/get-docker/)
    - [JetBrains (PyCharm) Docker Services](https://www.jetbrains.com/help/pycharm/docker.html) (optional)


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
   
!!! danger "Windows Known Issues"

      Git Handling Lines Ending: We recommend that you setup the following git configuration to avoid Windows to automatically add `\r\n` ending symbol line that Linux/MacOS do not support. To configure, use [this command](https://docs.github.com/en/get-started/getting-started-with-git/configuring-git-to-handle-line-endings#global-settings-for-line-endings):
        ```bash
        git config --global core.autocrlf true
        ```

!!! tip "Docker and Jetbrains"
    If you are using JetBrains, you should be able to leverage the `.run/` configurations at the root of the folder.
    They should be automatically detected by your Jetbrains IDE (e.g PyCharm) and you can run the tests from there.
    Make sure to edit the configuration to adapt to your use-case.

    Configs available:
    - `Scikit_longitudinal_ARM_architecture.run.xml`: If you are on an ARM architecture. Such as Macbook with Apple Silicon chips.
    - `Scikit_longitudinal_Intel_architecture.run.xml`: If you are on an Intel architecture. Such as most of the Windows and Linux machines or Macbook with Intel chips. 

!!! warning "Docker with Apple Silicon"
    If you are on an Apple Silicon chip, the current library is `x86_64` based. Therefore, you should configure Docker
    so that it runs on such architecture.  Be at the root of the project and run the following commands: 

    1. **Prepare [QUS](https://github.com/dbhi/qus) for Docker:**
        ```bash
        docker run --rm --privileged aptman/qus -- -r
        docker run --rm --privileged aptman/qus -s -- -p x86_64
        ```
    2. **Build the Docker Image:**
        ```bash
        docker buildx create --use
        docker buildx build --platform linux/amd64 -t scikit_longitudinal:latest .
        ```
    3. **Run the Docker Container:**
        ```bash
        docker run -it scikit_longitudinal:latest /bin/bash
        ```
    4. **Run the tests:**
        ```bash
        pytest scikit_longitudinal/ --cov=./ --cov-report=html --cov-config=.coveragerc --cov-report=html:htmlcov/scikit_longitudinal -s -vv --capture=no
        ```
## ⚙️ How To Build The Distribution Packages

To build the distribution packages for the project, follow these steps:

1. **Build the Distribution Packages:**
    - Run the following command to build the distribution packages:
      ```bash
      pdm build_dist
      ```

## 🫠 Not Satisfied Yet?

Further look in the `pyproject.toml` file for more commands that you can run to help you with the development process.

_🎉 Voilà! You are ready to contribute!_