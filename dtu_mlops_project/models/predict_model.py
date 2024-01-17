import os
import logging
import dotenv
import hydra
import torchaudio
import torch
import numpy as np


class PreprocPlusModel(torch.nn.Module):
    def __init__(self, model, preprocessing, path_to_model_checkpoint, device="cpu"):
        super(PreprocPlusModel, self).__init__()
        self.model = model
        self.preprocessing = preprocessing
        state_dict = torch.load(path_to_model_checkpoint, map_location=torch.device(device))["state_dict"]
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            x_preproc = self.preprocessing(x)
            logits = self.model(x_preproc)
            return x_preproc, logits

    def wav_to_yhat(self, path_to_waveform, read_true_class=False):
        waveform = torchaudio.load(path_to_waveform)[0].unsqueeze(dim=1)
        y_true = path_to_waveform.split('/')[-2] if read_true_class else None
        x_preproc, logits = self.forward(waveform)
        return {
            'waveform': waveform,
            'x_preproc': x_preproc,
            'logits': logits,
            'y_pred': logits.argmax().item(),
            'y_true': y_true,
        }


def get_ppm_from_config(cfg):
    model = hydra.utils.instantiate(cfg.model)
    preprocessing_list = hydra.utils.instantiate(cfg.datamodule.test_transforms)
    preprocessing = torch.nn.Sequential(*preprocessing_list)
    ppm = PreprocPlusModel(model, preprocessing, cfg.predict.filepath_model_checkpoint)
    return ppm


@hydra.main(version_base="1.3", config_path="../../configs", config_name="infer.yaml")
def predict_model(cfg):
    logger = logging.getLogger(__name__)
    logger.info(f"Working directory: {os.getcwd()}")

    # instantiate model+preprocessing module
    logger.info("Instantiating combined model+preprocessing module")
    ppm = get_ppm_from_config(cfg)

    # run inference
    path_to_waveform = os.environ["DATA_PATH"] + cfg.predict.filepath_dot_wav_audio
    results = ppm.wav_to_yhat(path_to_waveform=path_to_waveform, read_true_class=True)
    idx_to_class = np.load(cfg.idx_to_class).astype(str)
    predicted_class = idx_to_class[results["y_pred"]]
    logger.info(f"predicted_class: {predicted_class}     true class: {results['y_true']}")
    return predicted_class


if __name__ == "__main__":
    dotenv.load_dotenv(override=True)
    predict_model()
