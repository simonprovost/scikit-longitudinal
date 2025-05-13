---
hide:
  - navigation
---

## 🧪 Experiments with Scikit-Longitudinal & its various papers

Welcome to the Experiments guide for **Scikit-Longitudinal**! The following walks you through the process of conducting
experiments with the library, reproducing the various papers introduced with the library and make sure to test and verify
results.

Whether you're replicating experiments from our papers or designing your own, this guide will in a nuthsell help you
create data splits,
execute experiments on a High-Performance Computing (HPC) cluster using SLURM, and aggregate your results into a
single CSV file.

## 📰 Associated Papers

Below are the papers introduced with `Scikit-Longitudinal`, including their names, dataset sources, and current status.

=== "SepWav paper"

    **Paper Name**: Novel Ensemble Strategies for Wave-by-Wave Longitudinal Data Classification

    **Dataset Source**: https://www.elsa-project.ac.uk/

    !!! question "How to obtain the datasets (CSVs)?"
        The datasets are requested to be registered on the ELSA website. Therefore, we recommend you
        open an issue so that we can give you the exact steps by steps and provide you with the datasets safely.

    !!! warning "Status"
        Experiments completed and results reported in the paper.
        Paper is in preparation for submission.

=== "Lexicographical Trees paper"

    **Paper Name**: Longitudinal Classification Approached with Lexicographically Optimised Deep Forests and Gradient Boosting

    **Dataset Source**: https://www.elsa-project.ac.uk/

    !!! question "How to obtain the datasets (CSVs)?"
        The datasets are requested to be registered on the ELSA website. Therefore, we recommend you
        open an issue so that we can give you the exact steps by steps and provide you with the datasets safely.

    !!! warning "Status"
        Experiments completed and results reported in the paper.
        Paper is in preparation for submission.

=== "Library introduction paper"

    **Paper name**: Scikit-Longitudinal: A Machine Learning Library for Longitudinal Classification in Python

    **Dataset source**: No datasets. No experiments.

    !!! success "Status"
        Paper has been submitted to the Journal of Open Source Software (JOSS) and is currently under review.

        Follow the review at [https://github.com/openjournals/joss-reviews/issues/8189](https://github.com/openjournals/joss-reviews/issues/8189).

## References

- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
- [What is a HPC cluster?](https://en.wikipedia.org/wiki/Supercomputer)

!!! tip "Before You Begin"
    Ensure you have your dataset ready in a compatible format (e.g., CSV) and are familiar with the basic usage of
    Scikit-Longitudinal. Check out the [Getting Started](../getting-started.md) page if you need a refresher!

---

## Overview

The experiment workflow consists of three main steps:

1. **Create Cross-Validation Splits**: Generate training parallelized splits using `utils/create_splits.py` and any of
   the primitives in `experiment_launchers/`.
2. **Run Experiments**: Execute experiments on an HPC cluster with SLURM, utilizing generated splits.
3. **Report Results**: Aggregate experiment outputs into a single CSV file with `utils/experiment_results_reporter.py`.

We'll break down each step with detailed instructions, examples, and customization options.

---

## Step 1: Creating Cross-Validation Splits

### Purpose

To run experiments with cross-validation, you need to split your dataset into training and testing sets. The
`utils/create_splits.py` script automates this process, generating SLURM job scripts for each fold and scenario.

### How It Works

The script:

- [x] Defines a set of datasets and experiment scenarios (e.g., `sepwav_majority_voting`).
- [x] Creates fold-specific subdirectories under a base path (e.g., `experiments/scenarios/longitudinal_trees/10_folds`).
- [x] Generates SLURM `.sh` scripts for each fold, tailored to your experiment configuration.

### Customizing `create_splits.py`

Before running, modify the script to match your setup:

- **Number of Folds**: Adjust `num_folds` (default is 10).
  ```python
  num_folds = 5  # Change to your desired number of folds
  ```
- **Base Path**: Update `starting_path` to your preferred output directory.
  ```python
  starting_path = f"{os.getcwd()}/my_experiments/{num_folds}_folds"
  ```
- **Datasets**: Edit the `datasets` list to include your datasets.
  ```python
  datasets = ["my_dataset_1", "my_dataset_2"]
  ```
- **Parent Folders (Scenarios)**: Modify `parent_folders` to reflect your experiment types.
  ```python
  parent_folders = ["my_scenario"]
  ```
- **Target Columns**: Update `target_columns` dictionary with your dataset-specific target columns.
  ```python
  target_columns = {
      "my_dataset_1": "target_col_1",
      "my_dataset_2": "target_col_2"
  }
  ```
- **Dataset Paths**: Adjust the `dataset_path` construction if your data is stored differently.
  ```python
  dataset_path = f"/path/to/my/data/{dataset}.csv"
  ```

### Running the Script

Navigate to the `utils/` directory and execute:

```bash
python create_splits.py
```

or run it directly from the root of the repository:

```bash
python utils/create_splits.py
```

This generates SLURM scripts (e.g., `split_1.sh`) in subdirectories like
`my_experiments/5_folds/my_scenario/my_dataset_1/`.

!!! note "Output Structure"
    The script creates a folder hierarchy:
    ```
    my_experiments/5_folds/
    └── my_scenario/
        └── my_dataset_1/
            ├── split_1.sh
            ├── split_2.sh
            └── ...
    ```

---

## Step 2: Running Experiments on HPC with SLURM

### Purpose

Execute your experiments across multiple folds in parallel on an HPC cluster using SLURM, leveraging the scripts
generated in Step 1.

### Prerequisites

- Access to an HPC cluster with SLURM. Most universities and research institutions provide this, request so.
- Python environment configured with `Scikit-Longitudinal` dependencies. Follow the `getting-started` guide to set up your
  environment.

### Available Experiments

The `experiment_launchers/` directory contains scripts for two papers:

=== "Lexicographical Trees Paper"
    Algorithms candidates from the Lexicographical Trees paper:
      
      - `decision_tree.py`
      - `deep_forest.py`
      - `deep_forest_HPO.py`
      - `gradient_boosting.py`
      - `gradient_boosting_HPO.py`
      - `lexico_decision_tree.py`
      - `lexico_deep_forest.py`
      - `lexico_deep_forest_HPO.py`
      - `lexico_gradient_boosting.py`
      - `lexico_gradient_boosting_HPO.py`
      - `lexico_random_forest.py`
      - `nested_trees.py`
      - `random_forest.py`

=== "SepWav Paper"
    Algorithms from the SepWav paper:
      
      - `merwav_time_minus_random_forest.py`
      - `sepwav_decision_tree.py`
      - `sepwav_logistic_regression.py`
      - `sepwav_majority_voting.py`

### Submitting Jobs

1. **Navigate to a Scenario Folder**:
   ```bash
   cd my_experiments/5_folds/my_scenario/my_dataset_1/
   ```
2. **Submit All Fold Jobs**:
   Use `find` and `sbatch` to submit all `.sh` files:
   ```bash
   find . -name 'split_*.sh' | xargs -I {} sbatch {}
   ```
   This submits jobs for all folds (e.g., `split_1.sh` to `split_5.sh`) in parallel, given you have enough resources, or put in queue for later-on.
    
!!! tip "We recommend running one experiment prior running them all!"
    To avoid receiving too many emails, we recommend running one experiment first to check if everything is working as expected.

### Customizing SLURM Scripts

Each `.sh` file contains:

- **SLURM Directives**: Job name, output log, resource requests (e.g., memory, CPUs).
- **Environment Setup**: Activates your Python environment.
- **Experiment Command**: Calls an experiment launcher script with parameters.

Modify the SLURM template in `create_splits.py` if needed:

- **Resource Allocation**:
  ```bash
  #SBATCH --mem=4GB  # Increase memory if required
  #SBATCH --cpus-per-task=2  # Adjust CPU count
  ```
- **Email Notifications**:
  ```bash
  #SBATCH --mail-user="your.email@example.com"
  ```

!!! warning "Cluster-Specific Adjustments"
    Update the environment setup section (e.g., paths to `conda`, `pyenv`, or `pdm`) to match your HPC configuration.

---

## Step 3: Reporting Experiment Results

### Purpose

Aggregate fold-specific results into a single CSV file per experiment scenario, summarizing performance metrics.

### How It Works

The `utils/experiment_results_reporter.py` script:

- Merges `experiment_results.csv` files from each fold into a `merged_experiment_results.csv`.
- Computes the average of a specified metric across folds.
- Outputs a final CSV with datasets as rows and techniques as columns.

### Configuration

Create a JSON config file (e.g., `my_config.json`) based on examples like `long_tree_experiments_config.json`:

```json
{
  "root_placeholder": "MY_EXP",
  "datasets": [
    "my_dataset_1",
    "my_dataset_2"
  ],
  "techniques": [
    "my_scenario"
  ],
  "metric": "AUROC"
}
```

- **`root_placeholder`**: Prefix for experiment folders (e.g., `MY_EXP_my_dataset_1_my_scenario`).
- **`datasets`**: List of dataset names.
- **`techniques`**: List of scenario names.
- **`metric`**: Metric to average (e.g., `AUROC`, `Geometric Mean`).

### Running the Reporter

From the directory containing your experiment folders:

```bash
python utils/experiment_results_reporter.py --config my_config.json
```

This generates `final_merged_experiment_results_AUROC.csv`.

### Output Example

For the config above, the output might look like:

| Dataset | my_scenario |
|--------------|-------------|
| my_dataset_1 | 0.85 |
| my_dataset_2 | 0.82 |

---

## Putting It All Together

Here’s a complete example workflow:

1. **Modify `create_splits.py`**:
   ```python
   num_folds = 5
   starting_path = f"{os.getcwd()}/my_experiments/5_folds"
   datasets = ["stroke"]
   parent_folders = ["sepwav_majority_voting"]
   target_columns = {"stroke": "class_stroke_w8"}
   dataset_path = f"../data/{dataset}.csv"
   ```

2. **Generate Splits**:
   ```bash
   cd utils/
   python create_splits.py
   ```

3. **Run Experiments**:
   ```bash
   cd ../my_experiments/5_folds/sepwav_majority_voting/stroke/
   find . -name 'split_*.sh' | xargs -I {} sbatch {}
   ```

4. **Report Results**:
   Create `config.json`:
   ```json
   {
       "root_placeholder": "MY_EXP",
       "datasets": ["stroke"],
       "techniques": ["sepwav_majority_voting"],
       "metric": "Geometric Mean"
   }
   ```
   Then:
   ```bash
   cd ../../..
   python utils/experiment_results_reporter.py --config config.json
   ```

---

## Troubleshooting

- **Splits Not Created**: Check dataset paths and permissions in `create_splits.py`.
- **SLURM Errors**: Verify environment setup and resource requests in `.sh` files.
- **Reporting Fails**: Ensure all fold CSVs exist and the metric name matches a column header.

For further assistance, open an issue on our [GitHub](https://github.com/simonprovost/scikit-longitudinal/issues).

!!! question "Want to have our results already-ready-to investigate?"
    We save our results and all of the above steps for the various experiments in our papers. Feel free to request them
    by opening an issue.

Happy experimenting and reproducing our papers with `Scikit-Longitudinal`! 🎉