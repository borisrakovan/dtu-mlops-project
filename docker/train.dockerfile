FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt install -y ffmpeg && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/requirements.txt
COPY pyproject.toml /usr/src/app/pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

COPY dtu_mlops_project /usr/src/app/dtu_mlops_project
COPY data/ /usr/src/app/data/
COPY configs/ /usr/src/app/configs/

ENTRYPOINT ["python", "-u", "dtu_mlops_project/models/train_model.py"]
