import hydra
from hydra import compose
from dtu_mlops_project.constants import PROJECT_ROOT

hydra.core.global_hydra.GlobalHydra.instance().clear()
abs_config_dir = str(PROJECT_ROOT.joinpath('configs'))
hydra.initialize_config_dir(version_base="1.3", config_dir=abs_config_dir)
api_configs = compose(config_name='infer.yaml')
