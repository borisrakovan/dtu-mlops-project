import dotenv
from hydra import initialize, compose
from dtu_mlops_project.models.train_model import train_model

def test_train_model():
    dotenv.load_dotenv(override=True)
    with initialize(version_base="1.3", config_path="../../configs"):
        # config is relative to a module
        cfg = compose(config_name="train.yaml", overrides=[
            "trainer.overfit_batches=4",
            "logger.save_dir=../wandb_tests"])
        res = train_model(cfg)
        assert isinstance(res, dict)
        assert "train_loss" in res
        assert "test_loss" in res
