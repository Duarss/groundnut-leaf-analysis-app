# app/core/config.py
import os

# BASE_DIR = folder groundnut-backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")

    # Folder untuk model klasifikasi (sesuai struktur lama kamu)
    MODEL_DIR = os.path.join(BASE_DIR, "models", "classification")

    # Path ke file best_tuned_cfg.json yang dihasilkan dari training hyperparameter tuning
    BEST_TUNED_CFG_PATH = os.environ.get(
        "BEST_TUNED_CFG_PATH",
        os.path.join(MODEL_DIR, "best_tuned_cfg.json"),
    )

    # Path ke weights EfficientNet-B4 yang sudah kamu latih
    BEST_WEIGHTS_PATH = os.environ.get(
        "CLASSIFICATION_WEIGHTS_PATH",
        os.path.join(MODEL_DIR, "best_classification_model.weights.h5"),
    )

    # Path ke mapping kelas (dibuat oleh flow_from_directory saat training)
    CLASS_INDICES_PATH = os.environ.get(
        "CLASS_INDICES_PATH",
        os.path.join(MODEL_DIR, "classification_class_indices.json"),
    )

    # Ukuran input EfficientNet-B4 (harus sama dengan training)
    IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT", 380))
    IMG_WIDTH = int(os.environ.get("IMG_WIDTH", 380))

    # Jumlah kelas (sesuai NUM_CLASSES = 5 di skrip training) :contentReference[oaicite:1]{index=1}
    NUM_CLASSES = int(os.environ.get("NUM_CLASSES", 5))
