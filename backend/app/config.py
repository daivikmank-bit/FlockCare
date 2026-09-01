import os

MODEL_PATH: str = os.getenv("FLOCKCARE_MODEL_PATH", "ml/saved_models/flockcare_cnn.h5")
TFLITE_PATH: str = os.getenv("FLOCKCARE_TFLITE_PATH", "ml/saved_models/flockcare_cnn.tflite")
OOD_REF_PATH: str = os.getenv("FLOCKCARE_OOD_REF_PATH", "ml/saved_models/ood_reference.npz")
MAX_FILE_MB: int = int(os.getenv("FLOCKCARE_MAX_FILE_MB", "15"))
MIN_AUDIO_SECONDS: int = int(os.getenv("FLOCKCARE_MIN_AUDIO_SECONDS", "15"))
MIN_RMS_ENERGY: float = float(os.getenv("FLOCKCARE_MIN_RMS_ENERGY", "0.005"))

