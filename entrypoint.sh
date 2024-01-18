#!/bin/bash
set -e

# Echo command to let the user know what's happening
echo "Pulling data from DVC..."

dvc pull

# Unzip data, if necessary
mkdir -p /usr/src/app/data/processed
unzip /usr/src/app/data/raw/data.zip -d /usr/src/app/data/processed/

# Run the training script
exec python -u dtu_mlops_project/models/train_model.py "$@"