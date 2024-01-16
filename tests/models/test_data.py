import dotenv
import hydra
from hydra import initialize, compose
import logging
import torch

def test_train_preprocessing():
    """test for the training data preprocessing code.
        Input shape is .wav audio file.
        Output shape is inputshape of ResNet [B, 1, H, W] spectrogram.
    """
    dotenv.load_dotenv(override=True)

    logger = logging.getLogger(__name__)

    with initialize(version_base="1.3", config_path="../../configs"):
        # config is relative to a module
        cfg = compose(config_name="train.yaml")

        logger.info("Instantiating preprocessing")
        preprocessing_list = hydra.utils.instantiate(cfg.datamodule.train_transforms)

        preprocessing = torch.nn.Sequential(
            *preprocessing_list
        )

        batch_size = cfg.datamodule.batch_size

        x = torch.randn((batch_size, 1, 16000)) # inputsize of .wav audio file
        y = preprocessing(x)

        assert y.shape == (batch_size, 1, cfg.datamodule.train_transforms[2].n_mels, 63)


def test_test_preprocessing():
    """test for the test data preprocessing code.
        Input shape is .wav audio file.
        Output shape is inputshape of ResNet [B, 1, H, W] spectrogram.
        test data processing is the same as train, except it does not include frequency masking.
    """
    dotenv.load_dotenv(override=True)

    logger = logging.getLogger(__name__)

    with initialize(version_base="1.3", config_path="../../configs"):
        # config is relative to a module
        cfg = compose(config_name="train.yaml")

        logger.info("Instantiating preprocessing")
        preprocessing_list = hydra.utils.instantiate(cfg.datamodule.test_transforms) # no frequency masking

        preprocessing = torch.nn.Sequential(
            *preprocessing_list
        )

        batch_size = cfg.datamodule.batch_size

        x = torch.randn((batch_size, 1, 16000)) # inputsize of .wav audio file
        y = preprocessing(x)

        assert y.shape == (batch_size, 1, cfg.datamodule.test_transforms[1].n_mels, 63)
