import requests
import gradio as gr
from dtu_mlops_project.webapp.config import api_configs

url = f"http://localhost:{api_configs.api_port}/predict_audio"
headers = {'accept': 'application/json'}


def predict(file_path):
    if file_path is None:
        return ["No audio", None]
    # send audio to FastAPI and get prediction
    with open(file_path, 'rb') as f:
        files = {'data': (file_path, f, 'audio/wav')}
        response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        data = response.json()
        labels = {k: v/100 for k, v in data["logits"].items()}
        return [data["predicted_class"], labels]
    else:
        return [f"Error in prediction: {response.json()}", None]


# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath"),
    outputs=["text", "label"],
    title="Audio Prediction",
    description="Upload or record audio to get prediction"
)

demo.launch(server_port=api_configs.ui_port)
