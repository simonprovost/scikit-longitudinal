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

<img src="https://joss.theoj.org/papers/10.21105/joss.08481/status.svg" alt="DOI badge" >

</div>


---

# Wants to reproduce the SepWav Paper Experiments?

This folder contains the core code for running and managing experiments described in the SepWav paper. It supports reproducible experiments on longitudinal datasets using techniques like SepWav (Separate Waves) with various classifiers, handling class imbalance, and reporting metrics such as AUROC and Geometric Mean.

The setup uses nested cross-validation (outer folds for evaluation) and integrates with libraries like `scikit-longitudinal` for longitudinal data handling.

## Directory Structure

```
experiments/
├── experiment_engine.py              # Core engine for loading data, splitting, fitting models, and reporting results.
├── experiment_launchers/             # Scripts to launch specific experiments.
│   └── SepWav paper/                 # Launchers for paper-specific experiments.
│       ├── merwav_time_minus_random_forest.py   # Launcher for MerWav (Time-Minus) with Random Forest.
│       ├── sepwav_decision_tree.py              # Launcher for SepWav with stacking (Decision Tree meta-learner).
│       ├── sepwav_logistic_regression.py        # Launcher for SepWav with stacking (Logistic Regression meta-learner).
│       └── sepwav_majority_voting.py            # Launcher for SepWav with majority voting.
└── SLURM scripts/                            # All the scripts to run on SLURM servers.
|   ├── **scrips**.
└── utils/                            # Utility scripts for automation.
    ├── create_splits.py              # Generates SLURM job scripts for parallel fold execution.
    └── experiment_results_reporter.py # Merges fold results and computes average metrics.
```

## Prerequisites

- **Unpack The SLURM Scripts**: Ensure you have the SLURM scripts unpacked in the `experiments/SLURM scripts/` directory.
- **Python Version**: 3.9+ (tested with 3.9.8).
- **Environment Management**: Use `pdm` for dependency management (install via `pip install pdm`). Run `pdm install` in the project root to set up the virtual environment.
- **Key Libraries** (installed via `pdm`):
  - `pandas`, `scikit-learn`, `imbalanced-learn` (imblearn), `scikit-longitudinal`.
  - Highly Recommended: SLURM for HPC job submission.
- **Data**: Datasets should be CSV files (e.g., from ELSA dataset) with longitudinal features and a target column (e.g., `class_angina_w8`). Open an issue if your need the dataset, but you must register your group to ELSA first.
- **Export The experiments folder content to your HPC**: The experiments folder should be exported to your HPC environment, where you have all the dependencies installed.

## Run experiments in SLURM environment

### 1. Launching Experiments

We assume you are in SLURM env, with all dependencies installed.

To run experiments, use:

```bash
 find ./experiments/SLURM scripts/10_folds/<whatever_experiment_of_interest>/ -name 'split_*.sh' | xargs -I {} sbatch {}
```

## Using Utils

### 1. Generate SLURM Scripts for Parallel Folds (`create_splits.py`) — Yet already done!
Automates creating SLURM `.sh` files for running all folds on an HPC cluster.

- Edit `create_splits.py` to set `parent_folders` (e.g., ["sepwav_majority_voting"]), `datasets` (e.g., ["angina"]), and paths.
- Run: `python experiments/utils/create_splits.py`.
- Output: SLURM scripts in `experiments/scenarios/longitudinal_trees/10_folds/<technique>/<dataset>/split_X.sh`.
- Submit: `sbatch split_1.sh` (etc.) on your cluster.

**Customization**: Update `script_template` for email, resources, or paths.

### 2. Merge and Report Results (`experiment_results_reporter.py`)
After running all folds, merge CSVs and compute averages.

- Create a `config.json`:
  ```json
  {
    "root_placeholder": "SEPWAV_ / <<or any other PLACEHOLDER>>",
    "datasets": ["angina", "arthritis_core", "<<or any other dataset>>"],
    "techniques": ["sepwav_majority_voting", "sepwav_decision_tree", "<<or any other technique>>"],
    "metric": "AUROC" /* # Or any other metric you want to compute *\
  }
  ```
- Run: `python experiments/utils/experiment_results_reporter.py --config config.json`.
- Output: `merged_experiment_results.csv` per experiment folder, and `final_merged_experiment_results_AUROC.csv` (datasets as rows, techniques as columns with averages).

**Error Handling**: Scripts raise `ValueError` for invalid paths/params. Logs printed to console.

For paper-specific details, refer to the SepWav methodology in `scikit-longitudinal`. Questions? Open an issue or contact the authors.