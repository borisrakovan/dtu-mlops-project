SHELL=/bin/bash
CONDA_ENV=dtu_mlops_project


.PHONY: lint setup-data pull-data train build-image-train train-docker test test-coverage

venv:
	@echo "Activating conda environment"
	@source $(shell conda info --base)/etc/profile.d/conda.sh ;\
	conda activate $(CONDA_ENV)

build-image-train:
	docker build -t train:latest -f docker/train.dockerfile .

lint: venv
	ruff .

setup-data: venv
	@echo "Downloading datasets"
	python dtu_mlops_project/data/download_dataset.py data/raw


pull-data: venv
	@echo "Pulling data from DVC"
	@export $$(cat .gc-credentials.env | xargs)
	dvc pull
	mkdir -p data/processed
	unzip data/raw/data.zip -d data/processed/

train: venv
	python dtu_mlops_project/models/train_model.py

train-docker:
	docker run -it --rm --env-file .env -v $(shell pwd):/usr/src/app train:latest

make test: venv
	pytest tests/

make test-coverage: venv
	pytest tests/ --cov-config=.coveragerc --cov=dtu_mlops_project --cov-report=xml --cov-report=html
