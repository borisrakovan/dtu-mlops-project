.PHONY: lint data train

lint:
	poetry run ruff .

data:
	echo 'test' >> data/raw/test.txt && poetry run python src/data/make_dataset.py data/raw/test.txt data/processed/test.txt

train:
	poetry run python src/models/train_model.py