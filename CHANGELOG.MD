# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[Unreleased]: https://github.com/simonprovost/scikit-longitudinal/compare/v0.0.2...HEAD
[v0.0.2]: https://github.com/simonprovost/scikit-longitudinal/releases/tag/v0.0.2
[v0.0.1]: https://github.com/simonprovost/scikit-longitudinal/releases/tag/v0.0.1
