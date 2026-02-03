# app/services/history_service.py
from app.database.db import SessionLocal
from app.models.analysis_result import AnalysisResult


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
            items.append({
                "analysis_id": r.analysis_id,
                "created_at": r.created_at,  # biar route yang stringify (atau bisa iso di sini juga)
                "label": r.label,
                "confidence": r.confidence,
                "severity_pct": r.severity_pct,
            })

        return {
            "items": items,
            "limit": int(limit),
            "offset": int(offset),
        }
    finally:
        db.close()


def get_history_detail(analysis_id: str, client_id: str) -> dict:
    """
    WAJIB: client_id dikirim dari route (jangan pakai variable global / undefined).
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
            "created_at": row.created_at.isoformat() if row.created_at else None,
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
