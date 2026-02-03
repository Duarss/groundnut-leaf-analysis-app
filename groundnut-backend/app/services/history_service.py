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


def get_history_detail(client_id: str, analysis_id: str):
    db = SessionLocal()
    try:
        row = (
            db.query(AnalysisResult)
            .filter(AnalysisResult.client_id == client_id)
            .filter(AnalysisResult.analysis_id == analysis_id)
            .first()
        )
        return row
    finally:
        db.close()
