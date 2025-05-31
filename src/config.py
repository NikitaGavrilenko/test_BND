from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'

    # Основные параметры
    MODEL_NAME = 'yolov8x.pt'
    INPUT_VIDEO = 'crowd.mp4'
    CONF_THRESH = 0.4
    ENHANCE_IMAGE = True


config = Config()