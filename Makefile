.PHONY: lint data

lint:
	@echo "Running lint"
	@poetry run ruff .
	@echo "Lint complete"

data:
	@echo "Running data processing"
	@echo 'test' >> data/raw/test.txt && poetry run python src/data/make_dataset.py data/raw/test.txt data/processed/test.txt
	@echo "Data processing complete"