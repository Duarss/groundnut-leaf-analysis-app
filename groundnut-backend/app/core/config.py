# app/core/config.py
import os
from dotenv import load_dotenv

# Load .env sekali di awal aplikasi
load_dotenv()

def _env_bool(name: str, default: str = "0") -> bool:
    v = str(os.environ.get(name, default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

class Config:
    # =========================
    # APP / SECURITY
    # =========================
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")

    # BASE_DIR = folder groundnut-backend
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # =========================
    # TEMP STORAGE
    # =========================
    TEMP_DIR = os.environ.get(
        "TEMP_DIR",
        os.path.join(BASE_DIR, "tmp_uploads")
    )

    TEMP_TTL_SECONDS = int(os.environ.get("TEMP_TTL_SECONDS", 600))  # 10 menit
    TEMP_DELETE_AFTER_SEG = _env_bool("TEMP_DELETE_AFTER_SEG", "1")

    # Folder untuk simpan hasil permanen
    STORAGE_DIR = os.environ.get("STORAGE_DIR", os.path.join(BASE_DIR, "storage"))
    STORAGE_ANALYSIS_DIR = os.path.join(STORAGE_DIR, "analysis_results")

    # =========================
    # CLASSIFICATION
    # =========================
    CLSF_MODEL_DIR = os.path.join(BASE_DIR, "models", "classification")

    BEST_CLSF_TUNED_CFG_PATH = os.environ.get(
        "BEST_CLSF_TUNED_CFG_PATH",
        os.path.join(CLSF_MODEL_DIR, "best_tuned_cfg.json"),
    )

    BEST_CLSF_WEIGHTS_PATH = os.environ.get(
        "BEST_CLSF_WEIGHTS_PATH",
        os.path.join(CLSF_MODEL_DIR, "best_classification_model.weights.h5"),
    )

    CLSF_CLASS_INDICES_PATH = os.environ.get(
        "CLSF_CLASS_INDICES_PATH",
        os.path.join(CLSF_MODEL_DIR, "classification_class_indices.json"),
    )

    CLSF_IMG_H = int(os.environ.get("CLSF_IMG_H", 380))
    CLSF_IMG_W = int(os.environ.get("CLSF_IMG_W", 380))
    CLSF_NUM_CLASSES = int(os.environ.get("CLSF_NUM_CLASSES", 5))

    # =========================
    # SEGMENTATION
    # =========================
    SEG_MODEL_DIR = os.path.join(BASE_DIR, "models", "segmentation")

    BEST_SEG_TUNED_CFG_PATH = os.environ.get(
        "BEST_SEG_TUNED_CFG_PATH",
        os.path.join(SEG_MODEL_DIR, "best_tuned_cfg.json"),
    )

    BEST_SEG_WEIGHTS_PATH = os.environ.get(
        "BEST_SEG_WEIGHTS_PATH",
        os.path.join(SEG_MODEL_DIR, "best_segmentation_model.weights.h5"),
    )

    SEG_IMG_H = int(os.environ.get("SEG_IMG_H", 480))
    SEG_IMG_W = int(os.environ.get("SEG_IMG_W", 640))

    SEG_MASK_THRESHOLD = float(os.environ.get("SEG_MASK_THRESHOLD", 0.5))
    SEG_OVERLAY_ALPHA = float(os.environ.get("SEG_OVERLAY_ALPHA", 0.45))

    # =========================
    # SEVERITY (Leaf Mask)
    # =========================
    # Folder untuk model severity (leaf mask)
    SEV_MODEL_DIR = os.path.join(BASE_DIR, "models", "severity", "leaf_mask")

    # Path cfg + weights severity (U-Net small)
    BEST_SEV_CFG_PATH = os.environ.get(
        "BEST_SEV_CFG_PATH",
        os.path.join(SEV_MODEL_DIR, "best_tuned_cfg.json"),
    )

    BEST_SEV_WEIGHTS_PATH = os.environ.get(
        "BEST_SEV_WEIGHTS_PATH",
        os.path.join(SEV_MODEL_DIR, "best_severity_model.weights.h5"),
    )

    # Threshold binarisasi leaf mask (kamu minta 0.5)
    SEV_LEAF_MASK_THRESHOLD = float(os.environ.get("SEV_LEAF_MASK_THRESHOLD", 0.5))

    # Bins level FAO (5 level) dalam persen: 0-5-20-40-60-100
    # (boleh override via env, format: "0,5,20,40,60,100")
    SEV_FAO_BINS = os.environ.get("SEV_FAO_BINS", "0,5,20,40,60,100")

    # =========================
    # DATABASE (MySQL only)
    # =========================
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = os.environ.get("DB_PORT", "3306")  # default MySQL
    DB_NAME = os.environ.get("DB_NAME", "groundnut_db")
    DB_USER = os.environ.get("DB_USER", "root")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

    @classmethod
    def database_url(cls) -> str:
        """
        SQLAlchemy MySQL connection string
        """
        return (
            f"mysql+pymysql://{cls.DB_USER}:{cls.DB_PASSWORD}"
            f"@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
        )
