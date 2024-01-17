import os
import hydra
from hydra import compose

abs_config_dir = os.path.abspath("../../configs")
hydra.initialize_config_dir(version_base="1.3", config_dir=abs_config_dir)
api_configs = compose(config_name='infer.yaml')
