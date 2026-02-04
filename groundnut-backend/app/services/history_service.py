# app/services/history_service.py
from datetime import timezone
from app.database.db import SessionLocal
from app.models.analysis_result import AnalysisResult


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
