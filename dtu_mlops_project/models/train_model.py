import torch


def train_model():
    print("Training model...")
    print(f"PyTorch Version: {torch.__version__}")
    return "Model trained!"


if __name__ == "__main__":
    train_model()
