#!/bin/bash

export PDM_IN_ENV=in-project

conda init bash
source ~/.bashrc

eval $(pdm venv activate $PDM_IN_ENV)
