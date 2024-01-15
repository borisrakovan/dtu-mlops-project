from hydra import initialize, compose
from dtu_mlops_project.models.train_model import train_model

def test_train_model():
    with initialize(version_base="1.3", config_path="../../dtu_mlops_project/configs"):
        # config is relative to a module
        cfg = compose(config_name="train.yaml", overrides=[])
        res = train_model(cfg)
        assert isinstance(res, dict)
