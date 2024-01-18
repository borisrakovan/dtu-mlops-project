FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt install -y ffmpeg && apt install -y unzip && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt /usr/src/app/requirements.txt
COPY pyproject.toml /usr/src/app/pyproject.toml
COPY dtu_mlops_project /usr/src/app/dtu_mlops_project
COPY data/ /usr/src/app/data/
COPY configs/ /usr/src/app/configs/

# Copy the entrypoint script
COPY entrypoint.sh /usr/src/app/entrypoint.sh

# Give execution permissions to the script
RUN chmod +x /usr/src/app/entrypoint.sh

# Set the working directory in the container
WORKDIR /usr/src/app

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -e . --no-cache-dir

ENTRYPOINT ["python",  "-u", "dtu_mlops_project/models/train_model.py"]
