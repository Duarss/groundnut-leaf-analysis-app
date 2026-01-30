# app/utils/storage_io.py
import os
import shutil
from app.core.config import Config

def ensure_storage_dir():
    os.makedirs(Config.STORAGE_ANALYSIS_DIR, exist_ok=True)

def ensure_analysis_dir(analysis_id: str) -> str:
    """
    Buat folder khusus per analysis_id:
    storage/analysis_results/<analysis_id>/
    """
    ensure_storage_dir()
    d = os.path.join(Config.STORAGE_ANALYSIS_DIR, str(analysis_id))
    os.makedirs(d, exist_ok=True)
    return d

def persist_file(src_path: str, analysis_id: str, filename: str) -> str:
    """
    Copy file dari tmp_uploads ke storage permanen dalam folder analysis_id.
    filename contoh: 'orig.jpg', 'overlay.png'
    """
    analysis_dir = ensure_analysis_dir(analysis_id)
    dst = os.path.join(analysis_dir, filename)
    shutil.copy2(src_path, dst)
    return dst
