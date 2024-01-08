.PHONY: lint data train build-image-train


build-image-train:
	docker build -t train:latest -f docker/train.dockerfile .

lint:
	ruff .

data:
	echo 'test' >> data/raw/test.txt && python dtu_mlops_project/data/make_dataset.py data/raw/test.txt data/processed/test.txt

train:
	python dtu_mlops_project/models/train_model.py

train-docker:
	docker run -it --rm train:latest
