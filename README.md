dtu_mlops_project
==============================

Project repository of Group MLOps66 for course Machine Learning Operations @ DTU ([course website](https://skaftenicki.github.io/dtu_mlops/)). Group MLOps66 consists of Riccardo Miccini, Boris Rakovan, and Berend Schuit.

# Project description

1. **Project goal:**
The aim of this project is to build a DL pipeline that can classify audio data, by internally first converting it to spectrograms (similar to 2D image Tensor), and then treating the problem as an image classification task. The project will be based on the labeled audio data provided in the paper [Warden 2018](https://paperswithcode.com/paper/speech-commands-a-dataset-for-limited).
2. **Framework:**
Torchaudio will be used for loading the dataset, pre-processing and data augmentation. Subsequently, torchvision will be used to build a DL model to classify the 2D-Tensor spectrograms. As PyTorch third-party package, we will use [pytorch-image-models](https://github.com/huggingface/pytorch-image-models#models). A possible extended scope is to introduce noise in the torchaudio pre-processing pipeline using the third-party package [audiomentations](https://github.com/iver56/audiomentations) or possibly [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations).
3. **Data used:**
The dataset presented in the Warden 2018 paper is available as a [PyTorch standard dataset](https://pytorch.org/audio/main/generated/torchaudio.datasets.SPEECHCOMMANDS.html) in PyTorch audio.
4. **Deep learning model:**
Our aim is to design a DL model that is pre-trained on visual imagery data (e.g. ImageNet, COCO), which will then be finetuned on the described audio data.


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
