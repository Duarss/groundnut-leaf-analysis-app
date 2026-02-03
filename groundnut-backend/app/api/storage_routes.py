# app/routes/storage_routes.py
import os
from flask import Blueprint, request, send_file
from werkzeug.utils import safe_join

from app.core.config import Config
from app.database.db import SessionLocal
from app.models.analysis_result import AnalysisResult

storage_bp = Blueprint("storage_bp", __name__)


def _get_client_id() -> str:
    cid = (request.headers.get("X-Client-Id") or "").strip()
    if cid:
        return cid
    return (request.args.get("client_id") or "").strip()


def _authorize(db, analysis_id: str, client_id: str) -> bool:
    row = (
        db.query(AnalysisResult)
        .filter(AnalysisResult.analysis_id == analysis_id)
        .filter(AnalysisResult.client_id == client_id)
        .first()
    )
    return bool(row)


@storage_bp.get("/storage")
def get_storage_file_query():
    """
    URL final (karena blueprint prefix '/api'):
      /api/storage?path=<analysis_id>/orig.jpeg&client_id=<client_id>

    Opsional:
      /api/storage?path=<client_id>/<analysis_id>/orig.jpeg&client_id=<client_id>
    """
    client_id = _get_client_id()
    if not client_id:
        return {"error": "client_id wajib."}, 400

    rel_path = (request.args.get("path") or "").strip().lstrip("/\\")
    if not rel_path:
        return {"error": "path wajib"}, 400

    rel_path = rel_path.replace("\\", "/")
    parts = [p for p in rel_path.split("/") if p]
    if len(parts) < 2:
        return {"error": "path tidak valid"}, 400

    # Support 2 format:
    # NEW: "<client_id>/<analysis_id>/file"
    # OLD: "<analysis_id>/file"
    if len(parts) >= 3 and parts[0] == client_id:
        analysis_id = parts[1]
        rel_from_storage_root = rel_path  # "<client_id>/<analysis_id>/file"
        abs_path = safe_join(Config.STORAGE_ANALYSIS_DIR, rel_from_storage_root)
    else:
        analysis_id = parts[0]
        rel_under_client = rel_path  # "<analysis_id>/file"
        abs_path = safe_join(Config.STORAGE_ANALYSIS_DIR, client_id, rel_under_client)

    db = SessionLocal()
    try:
        if not _authorize(db, analysis_id, client_id):
            return {"error": "Data tidak ditemukan"}, 404
    finally:
        db.close()

    if not abs_path or not os.path.isfile(abs_path):
        return {"error": "File tidak ditemukan"}, 404

    return send_file(abs_path, as_attachment=False)


@storage_bp.get("/storage/<client_id>/<analysis_id>/<path:filename>")
def get_storage_file_path(client_id, analysis_id, filename):
    """
    URL final:
      /api/storage/<client_id>/<analysis_id>/orig.jpeg
    """
    cid = _get_client_id()
    if not cid:
        return {"error": "client_id wajib."}, 400
    if cid != client_id:
        return {"error": "client_id tidak cocok"}, 403

    db = SessionLocal()
    try:
        if not _authorize(db, analysis_id, cid):
            return {"error": "Data tidak ditemukan"}, 404
    finally:
        db.close()

    filename = (filename or "").replace("\\", "/")
    rel_path = f"{client_id}/{analysis_id}/{filename}".lstrip("/\\")
    abs_path = safe_join(Config.STORAGE_ANALYSIS_DIR, rel_path)
    if not abs_path or not os.path.isfile(abs_path):
        return {"error": "File tidak ditemukan"}, 404

    return send_file(abs_path, as_attachment=False)
