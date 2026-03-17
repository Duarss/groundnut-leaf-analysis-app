# app/services/save_service.py
import json
from datetime import datetime, timezone
from sqlalchemy.exc import SQLAlchemyError
from app.utils.temp_store import read_meta, find_image_path, delete_bundle
from app.utils.storage_io import persist_file
from app.database.db import SessionLocal
from app.models.analysis_result import AnalysisResult
from app.core.config import Config

def _meta_created_at_dt(meta: dict) -> datetime:
    v = meta.get("created_at", None)
    try:
        if isinstance(v, (int, float)) and v > 0:
            return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except Exception:
        pass
    return datetime.now(timezone.utc)


def save_analysis(analysis_id: str, client_id: str, delete_temp_after: bool | None = None) -> dict:
    client_id = (str(client_id).strip() if client_id is not None else "")
    if not client_id:
        raise ValueError("client_id wajib (untuk multi-user tanpa login).")

    meta = read_meta(analysis_id)

    cls = (meta.get("classification") or {})
    label = cls.get("label")
    conf = cls.get("confidence")
    probs = cls.get("probs")

    if label is None:
        raise ValueError("Meta classification tidak lengkap (label None).")

    orig_tmp_path = find_image_path(analysis_id)
    if not orig_tmp_path:
        raise FileNotFoundError("File gambar original tidak ditemukan di tmp_uploads.")

    ext = orig_tmp_path.rsplit(".", 1)[-1].lower()
    orig_saved_path = persist_file(orig_tmp_path, client_id, analysis_id, f"orig.{ext}")

    seg = meta.get("segmentation") or {}
    seg_enabled = bool(seg.get("enabled", False))

    overlay_tmp_path = seg.get("overlay_path")
    overlay_saved_path = None
    if seg_enabled and overlay_tmp_path:
        overlay_saved_path = persist_file(overlay_tmp_path, client_id, analysis_id, "overlay.png")

    sev = (seg.get("severity") or {}) if seg_enabled else {}
    severity_pct = sev.get("severity_pct", None)

    sad = sev.get("sad") or {}
    sad_scheme = sad.get("scheme")
    sad_class_index = sad.get("class_index")
    sad_midpoint_pct = sad.get("midpoint_pct")
    sad_range = sad.get("range_pct")
    sad_range_low = None
    sad_range_high = None
    if isinstance(sad_range, (list, tuple)) and len(sad_range) == 2:
        sad_range_low = sad_range[0]
        sad_range_high = sad_range[1]

    db = SessionLocal()
    try:
        row = AnalysisResult(
            analysis_id=analysis_id,
            created_at=_meta_created_at_dt(meta),
            client_id=client_id,

            orig_image_path=orig_saved_path,
            seg_overlay_path=overlay_saved_path,

            label=str(label),
            confidence=float(conf) if conf is not None else None,
            probs_json=json.dumps(probs, ensure_ascii=False) if probs is not None else None,

            seg_enabled=bool(seg_enabled),

            severity_pct=float(severity_pct) if severity_pct is not None else None,
            
            sad_scheme=str(sad_scheme) if sad_scheme else None,
            sad_class_index=int(sad_class_index) if sad_class_index is not None else None,
            sad_midpoint_pct=float(sad_midpoint_pct) if sad_midpoint_pct is not None else None,
            sad_range_low=float(sad_range_low) if sad_range_low is not None else None,
            sad_range_high=float(sad_range_high) if sad_range_high is not None else None,
        )
        db.add(row)
        db.commit()

    except SQLAlchemyError as e:
        db.rollback()
        raise RuntimeError(f"DB error: {e}") from e
    
    finally:
        db.close()

    if delete_temp_after is None:
        delete_temp_after = bool(Config.TEMP_DELETE_AFTER_SEG)
    if delete_temp_after:
        delete_bundle(analysis_id)

    return {
        "saved": True,
        "analysis_id": analysis_id,
        "client_id": client_id,
        "orig_image_path": orig_saved_path,
        "seg_enabled": seg_enabled,
        "seg_overlay_path": overlay_saved_path,
        "severity_pct": severity_pct,
        "sad": {
            "scheme": sad_scheme,
            "class_index": sad_class_index,
            "midpoint_pct": sad_midpoint_pct,
            "range_pct": sad_range,
        } if sad else None,
    }
