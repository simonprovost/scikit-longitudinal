#!/bin/bash

TEMP_PYTHON_PATH=/usr/local/bin/python
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    touch .env
fi
if grep -q "SKLONG_PYTHON_PATH" ".env"; then
    sed -i "s|SKLONG_PYTHON_PATH=.*|SKLONG_PYTHON_PATH=${TEMP_PYTHON_PATH}|" .env
else
    echo "SKLONG_PYTHON_PATH=${TEMP_PYTHON_PATH}" >> .env
fi

pdm config venv.backend conda
pdm use 3.9

pdm run setup_sklong
export PDM_IN_ENV=in-project

conda init bash
source ~/.bashrc

eval $(pdm venv activate $PDM_IN_ENV)
pdm run install_prod
pdm run install_dev
