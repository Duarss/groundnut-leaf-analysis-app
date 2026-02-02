# app/services/save_service.py
import json
import time
from sqlalchemy.exc import SQLAlchemyError

from app.utils.temp_store import read_meta, find_image_path
from app.utils.storage_io import persist_file
from app.database.db import SessionLocal
from app.models.analysis_result import AnalysisResult
from app.core.config import Config


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def save_analysis(analysis_id: str, delete_temp_after: bool | None = None) -> dict:
    """
    Save hasil final analisis:
    - Ambil meta JSON dari tmp_uploads
    - Copy original image (wajib) ke storage permanen
    - Copy overlay (opsional, jika segmentasi dilakukan)
    - Simpan 1 row ke analysis_results termasuk severity (opsional)
    """
    meta = read_meta(analysis_id)

    cls = (meta.get("classification") or {})
    label = cls.get("label")
    conf = cls.get("confidence")
    probs = cls.get("probs")

    if label is None:
        raise ValueError("Meta classification tidak lengkap (label None).")

    # original image wajib ada
    orig_tmp_path = find_image_path(analysis_id)
    if not orig_tmp_path:
        raise FileNotFoundError("File gambar original tidak ditemukan di tmp_uploads.")

    # original image
    ext = orig_tmp_path.rsplit(".", 1)[-1].lower()
    orig_saved_path = persist_file(orig_tmp_path, analysis_id, f"orig.{ext}")

    # segmentation
    seg = meta.get("segmentation") or {}
    overlay_tmp_path = seg.get("overlay_path")  # boleh None
    seg_enabled = bool(seg.get("enabled", False))

    overlay_saved_path = None
    if seg_enabled and overlay_tmp_path:
        overlay_saved_path = persist_file(overlay_tmp_path, analysis_id, "overlay.png")

    # severity (opsional)
    sev = seg.get("severity") or {}
    severity_pct = _safe_float(sev.get("severity_pct"))
    # suport beberapa kemungkinan struktur:
    # 1) sev["fao"]["level"]
    # 2) sev["fao_level"]
    fao = sev.get("fao") or {}
    severity_level = _safe_int(fao.get("level"))
    if severity_level is None:
        severity_level = _safe_int(sev.get("fao_level"))

    # default delete_temp_after pakai env kalau None
    if delete_temp_after is None:
        delete_temp_after = bool(getattr(Config, "TEMP_DELETE_AFTER_SEG", False))

    # insert DB
    db = SessionLocal()
    try:
        row = AnalysisResult(
            analysis_id=analysis_id,
            created_at=int(meta.get("created_at", 0)) or int(time.time()),
            orig_image_path=orig_saved_path,
            label=str(label),
            confidence=_safe_float(conf),
            probs_json=json.dumps(probs, ensure_ascii=False) if probs is not None else None,
            seg_enabled=bool(seg_enabled),
            seg_overlay_path=overlay_saved_path,

            # severity
            severity_pct=severity_pct,
            severity_fao_level=severity_level,
        )
        db.add(row)
        db.commit()

    except SQLAlchemyError as e:
        db.rollback()
        raise RuntimeError(f"DB error: {e}") from e
    finally:
        db.close()

    # bersihkan tmp bila diminta
    if delete_temp_after:
        from app.utils.temp_store import delete_bundle
        delete_bundle(analysis_id)

    return {
        "saved": True,
        "analysis_id": analysis_id,
        "orig_image_path": orig_saved_path,
        "seg_enabled": seg_enabled,
        "seg_overlay_path": overlay_saved_path,
        "severity_pct": severity_pct,
        "severity_fao_level": severity_level,
    }
