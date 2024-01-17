import os

import dotenv
import hydra
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from dtu_mlops_project.models.predict_model import PreprocPlusModel

dotenv.load_dotenv(override=True)


def generate_prediction_figure(path_to_model_checkpoint, path_to_waveform):

    ppm = PreprocPlusModel(path_to_model_checkpoint)
    results = ppm.wav_to_yhat(path_to_waveform=path_to_waveform)

    idx_to_class = np.load("data/processed/class_idx_export.npy").astype(str)
    predicted_class = idx_to_class[results["pred"]]
    waveform = results["waveform"]
    spectrogram = results['x_preproc'][0][0]

    # computing "confidence" of prediction.
    y_hat_activation = torch.nn.LogSoftmax(dim=1)
    y_hat = torch.exp(y_hat_activation(results["y_hat"])).detach().numpy()
    y_hat = (y_hat*100).flatten() # to percentages
    conf = str(np.round(y_hat[results["pred"]],3))



    fig = plt.figure(figsize=(12,10), tight_layout=True)
    gs = gridspec.GridSpec(5, 6)
    ax = fig.add_subplot(gs[0:3, 0:3])
    ax.set_title(f"waveform\ntrue_class: {results['y_true']}")
    ax.plot(waveform[0][0])

    ax = fig.add_subplot(gs[0:3,3:6])
    ax.set_title(f"spectrogram\npredicted_class: {predicted_class}     (conf: {conf}%)")
    ax.imshow(spectrogram, origin='lower')

    ax = fig.add_subplot(gs[3:, :])
    ax.bar(np.arange(len(y_hat)), y_hat)
    ax.set_xticks(np.arange(len(y_hat)), idx_to_class)
    ax.tick_params(axis='x', rotation=55)
    ax.set_ylabel('% "confidence"')
    plt.savefig("reports/figures/fig.png")


if __name__ == "__main__":

    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(config_name="train.yaml")

    generate_prediction_figure(path_to_model_checkpoint=cfg.predict.filepath_model_checkpoint,
                                path_to_waveform=os.environ["DATA_PATH"]+cfg.predict.filepath_dot_wav_audio
    )
