dtu_mlops_project
==============================

Coursework for the Machine Learning Operations class at DTU


# Setup

## Poetry

This project uses [poetry](https://python-poetry.org/) for dependency management. To install poetry, follow the instructions on the [poetry website](https://python-poetry.org/docs/#installation).

To create a virtual environment and install the dependencies, run

```bash
poetry install --no-root
```

You can verify that the installation was successful by running

```bash
make data
```

Visit the official poetry documentation for more information on how to [manage dependencies using poetry](https://python-poetry.org/docs/basic-usage/).


## Pre-commit

This project uses [pre-commit](https://pre-commit.com/) to run checks before committing. To install pre-commit, run

```bash
poetry run pre-commit install
```


# Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── pyproject.toml     <- Main configuration file for poetry
    ├── poetry.lock        <- Poetry lock file
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── tests              <- Unit tests

--------

<p><small>The project structure is adapted from the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
