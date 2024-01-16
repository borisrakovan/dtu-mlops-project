import dotenv
import hydra
from hydra import initialize, compose
import logging
import torch

def test_model():
    dotenv.load_dotenv(override=True)

    logger = logging.getLogger(__name__)

    with initialize(version_base="1.3", config_path="../../configs"):
        # config is relative to a module
        cfg = compose(config_name="train.yaml", overrides=[
            "trainer.overfit_batches=4"]) # no connection to WandB for github CI

        logger.info(f"Instantiating model <{cfg.model._target_}>")
        model = hydra.utils.instantiate(cfg.model)

        batch_size = cfg.datamodule.batch_size

        x = torch.randn((batch_size, 1, cfg.datamodule.train_transforms[2].n_mels, 63))
        # 63 is the time dimension, doesn't need to be correct.

        y = model(x)

        assert y.shape == (batch_size, 35)
