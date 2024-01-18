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
	@export $$(cat .gc-credentials.env | xargs) && dvc pull
	mkdir -p data/processed
	unzip data/raw/data.zip -d data/processed/

train: venv
	python dtu_mlops_project/models/train_model.py

predict:
	python dtu_mlops_project/models/predict_model.py

visualize:
	python dtu_mlops_project/visualization/visualize.py

train-docker-gpu:
	docker run -it --rm --user $(id -u):$(id -g) --env-file .env -v $(shell pwd):/usr/src/app --gpus all train:latest

train-docker:
	@GOOGLE_APPLICATION_CREDENTIALS=$$(grep GOOGLE_APPLICATION_CREDENTIALS .gc-credentials.env | cut -d '=' -f2) ; \
	docker run -it --rm \
		--user $(id -u):$(id -g) \
		--env-file .env \
		-e GOOGLE_APPLICATION_CREDENTIALS=/credentials.json \
		-v $$GOOGLE_APPLICATION_CREDENTIALS:/credentials.json:ro \
		-v $(shell pwd):/usr/src/app \
		train:latest

train-cloud:
	@DATE=$$(date +%Y%m%d-%H%M%S) ; \
	JOB_ID=$$(gcloud ai custom-jobs create \
		--region=europe-west1 \
		--display-name=train-run-$$DATE \
		--config=train_config_cpu.yml \
		--format="value(name)") ; \
	echo "Streaming logs for Job ID: $$JOB_ID" ; \
	gcloud ai custom-jobs stream-logs $$JOB_ID


test: venv
	pytest tests/

test-coverage: venv
	pytest tests/ --cov-config=.coveragerc --cov=dtu_mlops_project --cov-report=xml --cov-report=html

web-api: venv
	python dtu_mlops_project/webapp/main.py

web-gui: venv
	python dtu_mlops_project/webapp/gui.py

webapp: venv web-api web-gui
