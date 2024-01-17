import os
import logging
import dotenv
import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dtu_mlops_project.models.predict_model import get_ppm_from_config



@hydra.main(version_base="1.3", config_path="../../configs", config_name="infer.yaml")
def generate_prediction_figure(cfg):
    logger = logging.getLogger(__name__)
    logger.info(f"Working directory: {os.getcwd()}")

    # instantiate model+preprocessing module
    logger.info("Instantiating combined model+preprocessing module")
    ppm = get_ppm_from_config(cfg)

    # run inference
    logger.info("Running inference")
    path_to_waveform = os.environ["DATA_PATH"] + cfg.predict.filepath_dot_wav_audio
    results = ppm.wav_to_yhat(path_to_waveform=path_to_waveform, read_true_class=True)
    idx_to_class = np.load(cfg.idx_to_class).astype(str)
    predicted_class = idx_to_class[results["y_pred"]]
    waveform = results["waveform"]
    spectrogram = results['x_preproc'][0,0]

    # computing "confidence" of prediction.
    y_hat_activation = torch.nn.LogSoftmax(dim=1)
    y_hat = torch.exp(y_hat_activation(results["logits"]))
    y_hat = (y_hat*100).flatten() # to percentages
    conf = y_hat[results["y_pred"]]

    # plotting
    logger.info("Generating figure")
    fig = plt.figure(figsize=(12,10), tight_layout=True)
    gs = gridspec.GridSpec(5, 6)
    ax = fig.add_subplot(gs[0:3, 0:3])
    ax.set_title(f"waveform\ntrue_class: {results['y_true']}")
    ax.plot(waveform[0][0])

    ax = fig.add_subplot(gs[0:3,3:6])
    ax.set_title(f"spectrogram\npredicted_class: {predicted_class}     (conf: {conf:.2f}%)")
    ax.imshow(spectrogram, origin='lower')

    ax = fig.add_subplot(gs[3:, :])
    ax.bar(np.arange(len(y_hat)), y_hat)
    ax.set_xticks(np.arange(len(y_hat)), idx_to_class)
    ax.tick_params(axis='x', rotation=55)
    ax.set_ylabel('% "confidence"')
    plt.savefig("reports/figures/fig.png")


if __name__ == "__main__":
    dotenv.load_dotenv(override=True)
    generate_prediction_figure()
