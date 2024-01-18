import os
import logging
import subprocess
from pathlib import Path

import dotenv
import hydra
import lightning as L
from google.cloud import storage

from dtu_mlops_project.constants import PROJECT_ROOT


logger = logging.getLogger(__name__)

def prepare_data(bucket_name: str, source_blob_name: str, destination_file_name: str) -> None:
    """Downloads a blob from the bucket and unzips it."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        blob.download_to_filename(destination_file_name)
        logging.info(f"Blob {source_blob_name} downloaded to {destination_file_name}.")
    except Exception as e:
        logger.error(f"Error downloading blob: {e}")
        raise

    # Unzip data using subprocess
    try:
        processed_dir = str(PROJECT_ROOT / "data" / "processed")
        Path(processed_dir).mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        result = subprocess.run(["unzip", destination_file_name, "-d", processed_dir],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(result.stderr)
        logger.info(f"Data unzipped to {processed_dir}\n{result.stdout}")
    except Exception as e:
        logger.error(f"Error unzipping file: {e}")


def upload_to_gcs(bucket_name: str, source_file_name: str, destination_blob_name: str) -> None:
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    logger.info(f"File {source_file_name} uploaded to {destination_blob_name}.")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def train_model(cfg):
    logger = logging.getLogger(__name__)
    cwd = os.path.abspath(os.getcwd())
    logger.info(f"Current experiment directory: {cwd}")

    destination_file_name = PROJECT_ROOT / "data" / "raw" / "data.zip"

    # Check if data is already downloaded
    if not os.path.exists(destination_file_name):
        print(f"Data not found at {destination_file_name}. Downloading from GCS...")
        prepare_data(cfg.bucket_name, cfg.source_blob_name, destination_file_name)

    if cfg.get("seed"):
        logger.info(f"Setting seed to {cfg.seed}")
        L.seed_everything(cfg.seed, workers=True)

    logger.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    # logger.info("Instantiating callbacks...")
    # callbacks = instantiate_callbacks(cfg.get("callbacks"))
    callbacks = []

    logger.info(f"Instantiating logger <{cfg.logger._target_}>")
    pl_logger = hydra.utils.instantiate(cfg.logger)


    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=pl_logger)

    logger.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    ckpt_path = trainer.checkpoint_callback.best_model_path
    if cfg.get("test"):
        logger.info("Starting testing!")
        if ckpt_path == "":
            logger.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        logger.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}
    logger.info(f"Metrics: {metric_dict}")

    logger.info("Uploading model to GCS...")
    upload_to_gcs(cfg.bucket_name, ckpt_path, cfg.model_blob_name)


    # return optimized metric
    return metric_dict


if __name__ == "__main__":
    dotenv.load_dotenv(override=True)
    train_model()
