import os

import dotenv
import torchaudio
import torch
import hydra
import numpy as np

from dtu_mlops_project.models.model import SpeechSpectrogramsTransferLearning

dotenv.load_dotenv(override=True)


class PreprocPlusModel(torch.nn.Module):
    def __init__(self, path_to_model_checkpoint):
        super(PreprocPlusModel, self).__init__()

        with hydra.initialize(version_base="1.3", config_path="../../configs"):
            cfg = hydra.compose(config_name="train.yaml")

            preprocessing_list = hydra.utils.instantiate(cfg.datamodule.test_transforms)  # no frequency masking

            self.preprocessing = torch.nn.Sequential(
                *preprocessing_list
            )

        self.model = SpeechSpectrogramsTransferLearning.load_from_checkpoint(path_to_model_checkpoint).to(torch.device("cpu"))


    def forward(self, x):
        x_preproc = self.preprocessing(x)
        y_hat = self.model(x_preproc)
        return x_preproc, y_hat

    def wav_to_yhat(self, path_to_waveform):
        waveform = torchaudio.load(path_to_waveform)[0].unsqueeze(dim=1)
        y_true = path_to_waveform.split('/')[-2]
        x_preproc, y_hat = self.forward(waveform)

        return {
            'waveform': waveform,
            'x_preproc': x_preproc,
            'y_hat': y_hat,
            'pred': np.argmax(y_hat.detach().cpu().numpy()),
            'y_true': y_true,
        }




if __name__ == "__main__":

    path_to_model_checkpoint = "outputs/2024-01-16/20-31-40/dtu_mlops_project/nn38hdgq/checkpoints/epoch=2-step=7956.ckpt"
    # test_acc = 0.93412

    filename = "/SPEECHCMD/SpeechCommands/speech_commands_v0.02/eight/0a2b400e_nohash_2.wav"
    path_to_waveform = os.environ["DATA_PATH"]+filename


    ppm = PreprocPlusModel(path_to_model_checkpoint)
    results = ppm.wav_to_yhat(path_to_waveform=path_to_waveform)
    # print(results["x_preproc"].shape)

    idx_to_class = np.load("data/processed/class_idx_export.npy").astype(str)
    predicted_class = idx_to_class[results["pred"]]
    print(f"predicted_class: {predicted_class}     true class: {results['y_true']}")
