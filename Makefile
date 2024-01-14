SHELL=/bin/bash
CONDA_ENV=dtu_mlops_project


.PHONY: lint data train build-image-train train-docker test test-coverage

build-image-train:
	docker build -t train:latest -f docker/train.dockerfile .

lint:
	ruff .

data:
	@echo "Downloading datasets"
	@source $(shell conda info --base)/etc/profile.d/conda.sh ;\
	conda activate $(CONDA_ENV) ;\
	python src/data/download_dataset.py data/raw/

train:
	python dtu_mlops_project/models/train_model.py

train-docker:
	docker run -it --rm train:latest

make test:
	pytest tests/

make test-coverage:
	pytest --cov-config=.coveragerc --cov=dtu_mlops_project tests/ --cov-report=html
