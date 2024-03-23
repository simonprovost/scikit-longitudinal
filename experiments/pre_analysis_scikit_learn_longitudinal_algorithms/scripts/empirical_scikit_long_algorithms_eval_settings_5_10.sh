#!/bin/bash

#SBATCH --job-name=pre_analysis_sklong_5_10    # Job name
#SBATCH --output=pre_analysis_sklong_5_10.log  # Standard output and error log
#SBATCH --mail-type=BEGIN,END,FAIL           # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sgp28@kent.ac.uk # Where to send mail
#SBATCH --ntasks=11                     # Run on a single CPU
#SBATCH --cpus-per-task=1              # Number of CPU cores per task
#SBATCH --mem=15G                      # Job memory request
#SBATCH --partition=cpu

# Load miniconda
source ~/miniconda3/etc/profile.d/conda.sh

# Update PATH
export PATH="/home/arc/sgp28/miniconda3/bin/:$PATH" # For conda
export PATH="/home/arc/sgp28/.local/bin/:$PATH" # For pdm
export PATH="/home/arc/sgp28/.pyenv/bin/:$PATH" # For pyenv

# Make sure Python 3.9 is in use
pyenv local 3.9.8
pdm use 3.9

# Setup PDM in the environment
export PDM_IN_ENV=in-project
cd /home/arc/sgp28/scikit_longitudinal
eval $(pdm venv activate $PDM_IN_ENV)

# Echo commands to check environments
echo "Checking Conda Environment..."
which conda
conda info --envs | grep '*'

echo "Checking PDM Environment..."
which pdm
pdm info | grep 'Python Interpreter'

# Execute the python script
python experiments/pre_analysis_scikit_learn_longitudinal_algorithms/nested_cross_validation_scikit_long_algorithms_settings_5_10.py
