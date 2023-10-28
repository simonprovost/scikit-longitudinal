#!/bin/bash

pdm config venv.backend conda
pdm use 3.9

pdm run setup_project
export PDM_IN_ENV=in-project

conda init bash
source ~/.bashrc

eval $(pdm venv activate $PDM_IN_ENV)
pdm run install_project
