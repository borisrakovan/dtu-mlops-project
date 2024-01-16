from http import HTTPStatus
from fastapi import FastAPI
from fastapi import UploadFile, File

import torchaudio

app = FastAPI()


@app.get("/")
def root():
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.get("/cats/{cat_id}")
def show_cats(cat_id):
    response = {
        "cat_id": cat_id,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/predict_audio/")
async def predict_audio(data: UploadFile = File(...)):
    with open('tmp.wav', 'wb') as fp:
        content = await data.read()
        fp.write(content)
        fp.close()
    waveform, sample_rate = torchaudio.load('tmp.wav')

    response = {
        "shape": waveform.shape,
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response
