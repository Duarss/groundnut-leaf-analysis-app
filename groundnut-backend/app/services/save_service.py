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
    """
    meta["created_at"] sebelumnya epoch int.
    DB sekarang pakai TIMESTAMP/DateTime.
    """
    v = meta.get("created_at", None)
    try:
        if isinstance(v, (int, float)) and v > 0:
            return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except Exception:
        pass
    return datetime.now(timezone.utc)


def save_analysis(analysis_id: str, client_id: str, delete_temp_after: bool | None = None) -> dict:
    """
    Save hasil final analisis:
    - client_id WAJIB dikirim dari route (jangan ambil dari request di service)
    """
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

    # original image wajib ada
    orig_tmp_path = find_image_path(analysis_id)
    if not orig_tmp_path:
        raise FileNotFoundError("File gambar original tidak ditemukan di tmp_uploads.")

    # original image (persist)
    ext = orig_tmp_path.rsplit(".", 1)[-1].lower()
    orig_saved_path = persist_file(orig_tmp_path, client_id, analysis_id, f"orig.{ext}")

    # segmentation info
    seg = meta.get("segmentation") or {}
    seg_enabled = bool(seg.get("enabled", False))

    overlay_tmp_path = seg.get("overlay_path")  # boleh None
    overlay_saved_path = None
    if seg_enabled and overlay_tmp_path:
        overlay_saved_path = persist_file(overlay_tmp_path, client_id, analysis_id, "overlay.png")

    # severity info
    sev = seg.get("severity") or {}
    severity_pct = sev.get("severity_pct", None)

    severity_level = None
    fao = sev.get("fao") or {}
    if isinstance(fao, dict) and "level" in fao:
        severity_level = fao.get("level")

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
            severity_fao_level=int(severity_level) if severity_level is not None else None,
        )
        db.add(row)
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise RuntimeError(f"DB error: {e}") from e
    finally:
        db.close()

    # delete_temp_after: kalau None, pakai config
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
        "severity_fao_level": severity_level,
    }
