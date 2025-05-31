from pathlib import Path
import urllib.request
from src.config import config


def download_weights():
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
    Path(config.MODELS_DIR).mkdir(exist_ok=True)
    dest = config.MODELS_DIR / config.MODEL_NAME

    if not dest.exists():
        print("Downloading model weights...")
        urllib.request.urlretrieve(url, dest)
        print(f"Weights saved to: {dest}")


if __name__ == "__main__":
    download_weights()