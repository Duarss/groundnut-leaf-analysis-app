# app/services/history_service.py
import os
from datetime import timezone
from app.database.db import SessionLocal
from app.models.analysis_result import AnalysisResult
from app.core.config import Config


def _to_iso_utc(dt):
    """
    Pastikan output selalu ISO string.
    Kalau datetime dari DB naive (tanpa tzinfo), anggap UTC.
    """
    if dt is None:
        return None
    try:
        tz = getattr(dt, "tzinfo", None)
        if tz is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return str(dt)

def _safe_abs_path(rel_or_abs: str) -> str | None:
    """
    Ubah path yang tersimpan di DB menjadi absolute path yang aman di storage.
    DB kamu menyimpan path seperti "<client_id>/<analysis_id>/orig.jpg" (relatif).
    """
    if not rel_or_abs:
        return None

    p = str(rel_or_abs)

    # kalau sudah absolute, pakai apa adanya
    if os.path.isabs(p):
        abs_p = os.path.normpath(p)
    else:
        # join ke STORAGE_ANALYSIS_DIR
        base = os.path.abspath(Config.STORAGE_ANALYSIS_DIR)
        abs_p = os.path.normpath(os.path.join(base, p))

        # guard anti path traversal
        if not abs_p.startswith(base + os.sep) and abs_p != base:
            return None

    return abs_p


def _try_remove_file(path: str | None):
    if not path:
        return
    try:
        if os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
    except Exception:
        # jangan bikin delete gagal hanya karena file missing/locked
        pass


def _try_cleanup_dirs(path: str | None):
    """
    Bersihkan folder kosong: .../<client_id>/<analysis_id>/ (kalau kosong)
    """
    if not path:
        return
    try:
        d = os.path.dirname(path)
        # hapus folder analysis_id jika kosong
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)

        # coba hapus folder client_id kalau kosong (opsional, aman)
        parent = os.path.dirname(d)
        if os.path.isdir(parent) and not os.listdir(parent):
            os.rmdir(parent)
    except Exception:
        pass


def list_history(client_id: str, limit: int = 50, offset: int = 0):
    db = SessionLocal()
    try:
        q = (
            db.query(AnalysisResult)
            .filter(AnalysisResult.client_id == client_id)
            .order_by(AnalysisResult.created_at.desc())
            .offset(int(offset))
            .limit(int(limit))
        )
        rows = q.all()

        items = []
        for r in rows:
            items.append(
                {
                    "analysis_id": r.analysis_id,
                    "created_at": _to_iso_utc(r.created_at),
                    "label": r.label,
                    "confidence": r.confidence,
                    "seg_enabled": r.seg_enabled,
                    "severity_pct": r.severity_pct,
                    "severity_fao_level": r.severity_fao_level,
                }
            )

        return {"items": items, "limit": int(limit), "offset": int(offset)}
    finally:
        db.close()


def get_history_detail(analysis_id: str, client_id: str) -> dict:
    """
    WAJIB: client_id dikirim dari route.
    """
    if not client_id:
        raise ValueError("client_id wajib")

    db = SessionLocal()
    try:
        row = (
            db.query(AnalysisResult)
            .filter(AnalysisResult.analysis_id == analysis_id)
            .filter(AnalysisResult.client_id == client_id)
            .first()
        )
        if not row:
            return None

        return {
            "analysis_id": row.analysis_id,
            "client_id": row.client_id,
            "confidence": row.confidence,
            "created_at": _to_iso_utc(row.created_at),
            "label": row.label,
            "orig_image_path": row.orig_image_path,
            "probs_json": row.probs_json,
            "seg_enabled": row.seg_enabled,
            "seg_overlay_path": row.seg_overlay_path,
            "severity_fao_level": row.severity_fao_level,
            "severity_pct": row.severity_pct,
        }
    finally:
        db.close()

def delete_history_item(analysis_id: str, client_id: str, delete_files: bool = True) -> dict:
    """
    Hapus 1 item history:
    - validasi client_id (multi-user tanpa login)
    - hapus row DB
    - optional: hapus file orig + overlay di storage
    """
    if not client_id:
        raise ValueError("client_id wajib")

    db = SessionLocal()
    try:
        row = (
            db.query(AnalysisResult)
            .filter(AnalysisResult.analysis_id == analysis_id)
            .filter(AnalysisResult.client_id == client_id)
            .first()
        )
        if not row:
            return None

        orig_path = _safe_abs_path(row.orig_image_path)
        overlay_path = _safe_abs_path(row.seg_overlay_path)

        # delete DB row dulu (biar konsisten dari sisi API)
        db.delete(row)
        db.commit()

        if delete_files:
            _try_remove_file(orig_path)
            _try_remove_file(overlay_path)
            _try_cleanup_dirs(orig_path or overlay_path)

        return {"deleted": True, "analysis_id": analysis_id}
    finally:
        db.close()
