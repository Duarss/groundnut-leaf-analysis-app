# app/utils/storage_io.py
import os
import shutil
from pathlib import PurePosixPath

from app.core.config import Config

def ensure_storage_dir():
    os.makedirs(Config.STORAGE_ANALYSIS_DIR, exist_ok=True)

def ensure_analysis_dir(client_id: str, analysis_id: str) -> str:
    """
    storage/analysis_results/<client_id>/<analysis_id>/
    """
    ensure_storage_dir()
    d = os.path.join(Config.STORAGE_ANALYSIS_DIR, str(client_id), str(analysis_id))
    os.makedirs(d, exist_ok=True)
    return d

def persist_file(src_path: str, client_id: str, analysis_id: str, filename: str) -> str:
    """
    Copy file dari tmp_uploads -> storage permanen
    return: path RELATIF terhadap STORAGE_ANALYSIS_DIR/<client_id>
            contoh: "<analysis_id>/orig.jpg"
    """
    analysis_dir = ensure_analysis_dir(client_id, analysis_id)
    dst = os.path.join(analysis_dir, filename)
    shutil.copy2(src_path, dst)

    # return path relatif supaya gampang dipakai HistoryDetailPage.
    # WAJIB pakai slash '/' agar aman dipakai sebagai URL path di browser,
    # meskipun backend jalan di Windows.
    return str(PurePosixPath(str(analysis_id)) / filename)
