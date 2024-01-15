import os
import logging
import dotenv
import hydra
import lightning as L


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def train_model(cfg):
    logger = logging.getLogger(__name__)
    cwd = os.path.abspath(os.getcwd())
    logger.info(f"Current experiment directory: {cwd}")

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

    # logger.info("Instantiating loggers...")
    # pl_logger = instantiate_loggers(cfg.get("logger"))
    pl_logger = None

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=pl_logger)

    logger.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        logger.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            logger.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        logger.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}
    logger.info(f"Metrics: {metric_dict}")

    # return optimized metric
    return metric_dict


if __name__ == "__main__":
    dotenv.load_dotenv(override=True)
    train_model()
