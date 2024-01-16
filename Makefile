SHELL=/bin/bash
CONDA_ENV=dtu_mlops_project


.PHONY: lint setup-data pull-data train build-image-train train-docker test test-coverage

build-image-train:
	docker build -t train:latest -f docker/train.dockerfile .

lint:
	ruff .

setup-data:
	@echo "Downloading datasets"
	@source $(shell conda info --base)/etc/profile.d/conda.sh ;\
	conda activate $(CONDA_ENV) ;\
	python dtu_mlops_project/data/download_dataset.py data/raw


pull-data:
	@echo "Pulling data from DVC"
	@source $(shell conda info --base)/etc/profile.d/conda.sh ;\
	conda activate $(CONDA_ENV) ;\
	dvc pull

train:
	python dtu_mlops_project/models/train_model.py

train-docker:
	docker run -it --rm train:latest

make test:
	pytest tests/

make test-coverage:
	pytest tests/ --cov-config=.coveragerc --cov=dtu_mlops_project --cov-report=xml --cov-report=html
