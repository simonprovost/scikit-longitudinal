# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.0.4] - 2024-07-04 - First Public Release and Major Enhancements

### Added

- **Documentation**: Comprehensive new documentation with Material for MKDocs. This includes a detailed tutorial on understanding vectors of waves in longitudinal datasets, a contribution guide, an FAQ section, and complete API references for all estimators, preprocessors, data preparations, and the pipeline manager.
- **Docker Installation**: Added new Docker installation process.
- **Windows Support**: Windows is now supported via Docker.
- **New Classifiers/Regressors**: Introduced Lexico Deep Forest, Lexico Gradient Boosting, and Lexico Decision Tree Regressor.
- **PyPI Availability**: Scikit-Longitudinal is now available on PyPI.
- **Continuous Integration**: Integrated unit testing, documentation, and PyPI publishing within the CI pipeline.

### Improved

- **PDM Setup and Installation**: Enhanced setup and installation processes using PDM.
- **Testing Coverage**: Improved testing coverage, ensuring that nearly 90% of the library is tested.
- **Scikit-Lexicographical-Trees**: Extracted the lexicographical scikit-learn tree node splitting function into its own repository and published it to PyPI as Scikit-Lexicographical-Trees. This is now leveraged by our lexico-based estimators.
- **.env Management**: Improved management of environment variables.
- **Lexicographical Enhancements**: Integrated lexicographical enhancements of the waves vector within the variant of scikit-learn, scikit-lexicographical-trees, improving memory and time efficiency by handling algorithmic temporality directly in C++.

### To-Do

- **Docstrings Alignment**: Ensure that docstrings in the codebase align with the official documentation to avoid confusion.
- **Native Windows Compatibility**: Achieve Windows compatibility without relying on Docker (requires access to a Windows machine).
- **Future Enhancements**: Ongoing improvements and new features as they are identified.
- **Documentation examples**: Add examples to the documentation to help users understand how to use the library with Jupyter notebooks.

## [v0.0.3] - 2023-10-31 - Usability, Maintainability, and Compliance Enhancements

### Added

- Features Group Missing Waves Handling: Introduced mechanisms for gracefully handling missing waves in features groups.
- Readiness Descriptions: New readiness indicators provide detailed descriptions of temporal data management across the library.
- Auto-Sklong Compliance: The library is now compliant with Auto-Sklong standards.
- Package Management Transition: Switched from Poetry to PDM for improved package and dependency management.
- Docker Support: Linux-based Docker environment setup for streamlined installation and deployment.
- Platform Testing: Library is tested on both Mac and Linux, with Windows support nearing completion.
- Documentation: Comprehensive version 0.0.1 of the documentation is available on GitHub Pages.
- Pipeline Manager: Refactored the pipeline into a more maintainable and flexible pipeline manager.
- CFS Classes Refactoring: Separated CFS and CFS Per Group algorithms into distinct classes for better management.

### Removed

- Irrelevant Scripts: Removed scripts related to visualizations not core to the library's functionality.
- Experiments Branch: Moved all experiment-related codes to a dedicated `Experiments` branch.

## [v0.0.2] - 2023-05-17 - Enhanced Longitudinal Analysis and Parallelization Features

### Added

- Implementation and validation of the three CFS Per Group Nested Tree and LexicoRF algorithms.
- Parallelization enhancements where possible.
- Longitudinal dataset handler for access to non-longitudinal features, longitudinal features group, etc.
- Longitudinal pipeline for longitudinal-based algorithms that pass features group onto each step of the pipeline.
- Comprehensive documentation and extensive test coverage (>95% of the codebase).
- Git hooks and other tools for long-term project use.
- An improved version of the CFS per Group algorithm (version two) based on the paper's concept level.
- Updated README file.

## [v0.0.1] - 2023-03-27 - Initial Release

### Added

- Initial setup of the Poetry Python project with robust type-checking.
- Integration of linting tools: pylint, flake8, pre-commit, black, and isort.
- Correlation-based Feature Selection (CFS) algorithm with improved typing and testing.
- CFS per Group for Longitudinal Data: Python implementation with parallelism for better performance.

[Unreleased]: https://github.com/simonprovost/scikit-longitudinal/compare/v0.0.3...HEAD
[v0.0.4]: https://github.com/simonprovost/scikit-longitudinal/releases/tag/v0.0.4
[v0.0.3]: https://github.com/simonprovost/scikit-longitudinal/releases/tag/v0.0.3
[v0.0.2]: https://github.com/simonprovost/scikit-longitudinal/releases/tag/v0.0.2
[v0.0.1]: https://github.com/simonprovost/scikit-longitudinal/releases/tag/v0.0.1