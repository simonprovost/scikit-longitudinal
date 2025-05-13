import os

# Define the number of folds
num_folds = 10

# Starting Path
starting_path = f"{os.getcwd()}/experiments/scenarios/longitudinal_trees/{num_folds}_folds"

# Define the parent folders and datasets
parent_folders = [
    "sepwav_majority_voting",
]
datasets = [
    "angina",
    "arthritis",
    "heartattack",
    "dementia",
    "heartattack_core",
    "hbp_core",
    "cataract_core",
    "angina_core",
    "diabetes",
    "osteoporosis_core",
    "stroke_core",
    "dementia_core",
    "cataract",
    "diabetes_core",
    "stroke",
    "parkinsons",
    "parkinsons_core",
    "hbp",
    "arthritis_core",
    "osteoporosis",
]

# SLURM script template
script_template = """#!/bin/bash

# ========================================
# SLURM Configuration
# ========================================
#SBATCH --job-name="{job_name}"
#SBATCH --output="{output_name}"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user="enter_your_email@gmail.com"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --partition=cpu

# ========================================
# Experiment Variables
# ========================================
N_OUTER_SPLITS=10

DATASET_PATH="{dataset_path}"
TARGET_COLUMN="{target_column}"
FOLD_NUMBER={fold_number}
EXPORT_NAME="{export_name}"

# ========================================
# Environment Setup (NO NEED TO MODIFY)
# ========================================
source ~/miniconda3/etc/profile.d/conda.sh
export PATH="/home/arc/sgp28/miniconda3/bin/:$PATH"
export PATH="/home/arc/sgp28/.local/bin/:$PATH"
export PATH="/home/arc/sgp28/.pyenv/bin/:$PATH"
pyenv local 3.9.8
pdm use 3.9
export PDM_IN_ENV=in-project
cd /home/arc/sgp28/Auto-Sklong
eval $(pdm venv activate $PDM_IN_ENV)

# ========================================
# Run Experiment (NO NEED TO MODIFY)
# ========================================
python ./experiments/experiment_launchers/{parent_folder}.py --dataset_path $DATASET_PATH --target_column $TARGET_COLUMN --fold_number $FOLD_NUMBER --export_name $EXPORT_NAME --n_outer_splits $N_OUTER_SPLITS
"""

# Define target columns for each dataset (assuming they follow a similar naming convention)
target_columns = {
    "angina": "class_angina_w8",
    "arthritis": "class_arthritis_w8",
    "heartattack": "class_heartattack_w8",
    "dementia": "class_dementia_w8",
    "heartattack_core": "class_heartattack_w8",
    "hbp_core": "class_hbp_w8",
    "cataract_core": "class_cataract_w8",
    "angina_core": "class_angina_w8",
    "diabetes": "class_diabetes_w8",
    "osteoporosis_core": "class_osteoporosis_w8",
    "stroke_core": "class_stroke_w8",
    "dementia_core": "class_dementia_w8",
    "cataract": "class_cataract_w8",
    "diabetes_core": "class_diabetes_w8",
    "stroke": "class_stroke_w8",
    "parkinsons": "class_parkinsons_w8",
    "parkinsons_core": "class_parkinsons_w8",
    "hbp": "class_hbp_w8",
    "arthritis_core": "class_arthritis_w8",
    "osteoporosis": "class_osteoporosis_w8",
}

# Iterate over each parent folder
for parent_folder in parent_folders:
    # Iterate over each dataset folder
    for dataset in datasets:
        dataset_folder = os.path.join(starting_path, parent_folder, dataset)

        # Make sure the dataset folder exists
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

        # Define the dataset path
        if "core" in dataset:
            scope = "core"
            dataset_name = dataset.split("_")[0]
        else:
            scope = "nurse"
            dataset_name = dataset

        dataset_path = f"../scikit_longitudinal/data/elsa/{scope}/csv/{dataset_name}_dataset.csv" # Modify this path as needed
        target_column = target_columns.get(dataset, None)
        if target_column is None:
            raise ValueError(f"Target column not found for dataset {dataset}")
        export_name = f"LONG_TREES_{dataset.upper()}_{parent_folder.upper()}" # It is not big deal for SEPWAV.

        # Generate scripts for each fold
        for fold_number in range(1, num_folds + 1):
            print(f"Generating script for {dataset_folder} - Fold {fold_number}...")
            script_filename = os.path.join(dataset_folder, f"split_{fold_number}.sh")

            job_name = f"LONG_TREES_{dataset.upper()}_{parent_folder.upper()}_split_{fold_number}"
            output_name = f"LONG_TREES_{dataset.upper()}_{parent_folder.upper()}_split_{fold_number}.log"
            script_content = script_template.format(
                job_name=job_name,
                output_name=output_name,
                dataset_path=dataset_path,
                target_column=target_column,
                fold_number=fold_number,
                export_name=export_name,
                parent_folder=parent_folder
            )

            # Write the script to a file
            with open(script_filename, 'w') as script_file:
                script_file.write(script_content)

print("Scripts generated successfully.")