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
    Copy file dari tmp_uploads -> storage permanen.

    return: path RELATIF terhadap Config.STORAGE_ANALYSIS_DIR
            contoh: "<client_id>/<analysis_id>/orig.jpeg"
    NOTE: sengaja pakai POSIX path ('/') agar aman untuk URL di browser
    walaupun backend berjalan di Windows.
    """
    analysis_dir = ensure_analysis_dir(client_id, analysis_id)
    dst = os.path.join(analysis_dir, filename)
    shutil.copy2(src_path, dst)

    return str(PurePosixPath(str(client_id)) / str(analysis_id) / filename)
