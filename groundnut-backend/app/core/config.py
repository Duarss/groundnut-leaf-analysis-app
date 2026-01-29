# app/core/config.py
import os

# BASE_DIR = folder groundnut-backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")

    # ============ TEMP STORAGE (upload sekali, dipakai ulang) ============
    TEMP_DIR = os.environ.get("TEMP_DIR", os.path.join(BASE_DIR, "tmp_uploads"))
    TEMP_TTL_SECONDS = int(os.environ.get("TEMP_TTL_SECONDS", 10 * 60))  # 10 menit
    TEMP_DELETE_AFTER_SEG = bool(int(os.environ.get("TEMP_DELETE_AFTER_SEG", "0")))

    # =========================
    # CLASSIFICATION
    # =========================
    # Folder untuk model klasifikasi
    CLSF_MODEL_DIR = os.path.join(BASE_DIR, "models", "classification")

    # Path ke file best_tuned_cfg.json yang dihasilkan dari training hyperparameter tuning
    BEST_CLSF_TUNED_CFG_PATH = os.environ.get(
        "BEST_CLSF_TUNED_CFG_PATH",
        os.path.join(CLSF_MODEL_DIR, "best_tuned_cfg.json"),
    )

    # Path ke weights EfficientNet-B4 yang sudah kamu latih
    BEST_CLSF_WEIGHTS_PATH = os.environ.get(
        "BEST_CLSF_WEIGHTS_PATH",
        os.path.join(CLSF_MODEL_DIR, "best_classification_model.weights.h5"),
    )

    # Path ke mapping kelas (dibuat oleh flow_from_directory saat training)
    CLSF_CLASS_INDICES_PATH = os.environ.get(
        "CLSF_CLASS_INDICES_PATH",
        os.path.join(CLSF_MODEL_DIR, "classification_class_indices.json"),
    )

    # Ukuran input EfficientNet-B4
    CLSF_IMG_H = int(os.environ.get("CLSF_IMG_H", 380))
    CLSF_IMG_W = int(os.environ.get("CLSF_IMG_W", 380))

    # Jumlah kelas (sesuai NUM_CLASSES = 5 di skrip training) :contentReference[oaicite:1]{index=1}
    CLSF_NUM_CLASSES = int(os.environ.get("CLSF_NUM_CLASSES", 5))

    # =========================
    # SEGMENTATION
    # =========================
    # Folder untuk model segmentasi
    SEG_MODEL_DIR = os.path.join(BASE_DIR, "models", "segmentation")

    # Path ke file best_tuned_cfg.json yang dihasilkan dari training hyperparameter tuning
    BEST_SEG_TUNED_CFG_PATH = os.environ.get(
        "BEST_SEG_TUNED_CFG_PATH",
        os.path.join(SEG_MODEL_DIR, "best_tuned_cfg.json"),
    )

    # Path ke weights U-Net dengan encoder EfficientNet-B0
    BEST_SEG_WEIGHTS_PATH = os.environ.get(
        "BEST_SEG_WEIGHTS_PATH",
        os.path.join(SEG_MODEL_DIR, "best_segmentation_model.weights.h5"),
    )

    # Ukuran input U-Net
    SEG_IMG_H = int(os.environ.get("SEG_IMG_H", 480))
    SEG_IMG_W = int(os.environ.get("SEG_IMG_W", 640))

    # Threshold binarisasi mask output segmentasi
    SEG_MASK_THRESHOLD = float(os.environ.get("SEG_MASK_THRESHOLD", 0.5))
