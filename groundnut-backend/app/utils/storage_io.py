# app/utils/storage_io.py
import os
import shutil
from app.core.config import Config

def ensure_storage_dir():
    os.makedirs(Config.STORAGE_ANALYSIS_DIR, exist_ok=True)

def persist_file(src_path: str, analysis_id: str, suffix: str) -> str:
    """
    Copy file dari tmp_uploads ke storage permanen.
    suffix contoh: "orig.jpg" atau "overlay.png"
    """
    ensure_storage_dir()
    dst = os.path.join(Config.STORAGE_ANALYSIS_DIR, f"{analysis_id}_{suffix}")
    shutil.copy2(src_path, dst)
    return dst
