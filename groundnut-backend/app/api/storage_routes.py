# app/routes/storage_routes.py
import os
from flask import Blueprint, request, send_file

from werkzeug.utils import safe_join
from app.core.config import Config
from app.database.db import SessionLocal
from app.models.analysis_result import AnalysisResult

storage_bp = Blueprint("storage_bp", __name__)


def _get_client_id() -> str:
    # img tag tidak bisa kirim header kadang, jadi dukung query juga
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


@storage_bp.get("/api/storage")
def get_storage_file_query():
    """
    Serve file storage via query:
      /api/storage?path=<client_id>/<analysis_id>/orig.jpg&client_id=...
      /api/storage?path=<client_id>/<analysis_id>/overlay.png&client_id=...

    path disimpan di DB biasanya relpath: "<analysis_id>/orig.jpg"
    """
    client_id = _get_client_id()
    if not client_id:
        return {"error": "client_id wajib (untuk multi-user tanpa login)."}, 400

    # Terima path dari FE/DB yang kadang pakai backslash (Windows).
    # Normalisasi ke slash agar parsing konsisten dan aman.
    rel_path = (request.args.get("path") or "").strip().lstrip("/\\")
    rel_path = rel_path.replace("\\", "/")
    if not rel_path:
        return {"error": "path wajib"}, 400

    # Dukungan 2 format rel_path:
    # 1) "<analysis_id>/orig.jpg" (ini yang disimpan di DB oleh persist_file)
    # 2) "<client_id>/<analysis_id>/orig.jpg" (opsional / legacy)
    parts = [p for p in rel_path.split("/") if p]
    if len(parts) < 2:
        return {"error": "path tidak valid"}, 400

    if parts[0] == client_id:
        # format 2) path sudah menyertakan client_id
        analysis_id = parts[1]
        rel_under_client = "/".join(parts[1:])
    else:
        # format 1)
        analysis_id = parts[0]
        rel_under_client = rel_path

    db = SessionLocal()
    try:
        if not _authorize(db, analysis_id, client_id):
            return {"error": "Data tidak ditemukan"}, 404
    finally:
        db.close()

    # root storage: storage/analysis_results/<client_id>/<analysis_id>/...
    abs_path = safe_join(Config.STORAGE_ANALYSIS_DIR, client_id, rel_under_client)
    if not abs_path or not os.path.isfile(abs_path):
        return {"error": "File tidak ditemukan"}, 404

    return send_file(abs_path, as_attachment=False)


@storage_bp.get("/api/storage/<client_id>/<analysis_id>/<path:filename>")
def get_storage_file_path(client_id, analysis_id, filename):
    """
    Alternate route:
      /api/storage/<client_id>/<analysis_id>/orig.jpg
    """
    # Jika header/query client_id dikirim, wajib match dengan path param.
    cid = _get_client_id()
    if not cid:
        return {"error": "client_id wajib (untuk multi-user tanpa login)."}, 400
    if cid and cid != client_id:
        return {"error": "client_id tidak cocok"}, 403
    client_id = cid

    db = SessionLocal()
    try:
        if not _authorize(db, analysis_id, client_id):
            return {"error": "Data tidak ditemukan"}, 404
    finally:
        db.close()

    # filename bisa mengandung backslash kalau datang dari Windows; normalisasi.
    filename = (filename or "").replace("\\", "/")
    rel_path = f"{analysis_id}/{filename}".lstrip("/\\")
    abs_path = safe_join(Config.STORAGE_ANALYSIS_DIR, client_id, rel_path)
    if not abs_path or not os.path.isfile(abs_path):
        return {"error": "File tidak ditemukan"}, 404

    return send_file(abs_path, as_attachment=False)
