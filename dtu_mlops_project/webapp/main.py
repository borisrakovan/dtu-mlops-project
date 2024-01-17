import os
import torch
import numpy as np
from omegaconf import OmegaConf
from http import HTTPStatus
from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.responses import PlainTextResponse
from dtu_mlops_project.constants import PROJECT_ROOT
from dtu_mlops_project.webapp.config import api_configs
from dtu_mlops_project.models.predict_model import get_ppm_from_config

os.chdir(PROJECT_ROOT)
app = FastAPI()
ppm = get_ppm_from_config(api_configs)
idx_to_class = np.load(api_configs.idx_to_class).astype(str)


@app.get("/")
def root():
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.get("/get_model_descr", response_class=PlainTextResponse)
def get_model_descr():
    response = f"{ppm}"
    return response


@app.get("/get_configs")
def get_configs():
    response = OmegaConf.to_container(api_configs, resolve=True)
    return response


@app.post("/predict_audio/")
async def predict_audio(data: UploadFile = File(...)):
    results = ppm.wav_to_yhat(path_to_waveform=data.file, read_true_class=False)
    predicted_class = idx_to_class[results["y_pred"]]

    # computing "confidence" of prediction.
    y_hat_activation = torch.nn.LogSoftmax(dim=1)
    y_hat = torch.exp(y_hat_activation(results["logits"]))
    y_hat = (y_hat*100).flatten().numpy().tolist()

    response = {
        "predicted_class": predicted_class,
        "predicted_class_idx": results["y_pred"],
        "logits": {k: v for k, v in zip(idx_to_class, y_hat)}
    }
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=api_configs.api_port)
