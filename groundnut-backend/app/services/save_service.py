# app/services/save_service.py
import json
from sqlalchemy.exc import SQLAlchemyError

from app.utils.temp_store import read_meta, find_image_path
from app.utils.storage_io import persist_file
from app.database.db import SessionLocal
from app.models.analysis_result import AnalysisResult
from app.core.config import Config


def save_analysis(analysis_id: str, delete_temp_after: bool = False) -> dict:
    """
    Save hasil final analisis:
    - Ambil meta JSON dari temp_store
    - Copy original image (wajib) ke storage permanen
    - Copy overlay (opsional, jika ada dan segmentasi dilakukan)
    - Insert 1 row ke analysis_results
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

    # copy original → storage permanen
    # suffix: pakai ekstensi dari tmp
    ext = orig_tmp_path.rsplit(".", 1)[-1].lower()
    orig_saved_path = persist_file(orig_tmp_path, analysis_id, f"orig.{ext}")

    # overlay opsional: tergantung flow segmentasi kamu
    # asumsikan meta["segmentation"]["overlay_path"] menyimpan path file overlay di tmp_uploads
    seg = meta.get("segmentation") or {}
    overlay_tmp_path = seg.get("overlay_path")  # boleh None
    seg_enabled = bool(seg.get("enabled", False))

    overlay_saved_path = None
    if seg_enabled and overlay_tmp_path:
        overlay_saved_path = persist_file(overlay_tmp_path, analysis_id, "overlay.png")

    # insert DB
    db = SessionLocal()
    try:
        row = AnalysisResult(
            analysis_id=analysis_id,
            created_at=int(meta.get("created_at", 0)) or int(__import__("time").time()),
            orig_image_path=orig_saved_path,
            label=str(label),
            confidence=float(conf) if conf is not None else None,
            probs_json=json.dumps(probs, ensure_ascii=False) if probs is not None else None,
            seg_enabled=bool(seg_enabled),
            seg_overlay_path=overlay_saved_path,
        )
        db.add(row)
        db.commit()

    except SQLAlchemyError as e:
        db.rollback()
        raise RuntimeError(f"DB error: {e}") from e
    finally:
        db.close()

    # opsional: bersihkan tmp (kalau kamu mau)
    if delete_temp_after:
        from app.utils.temp_store import delete_bundle
        delete_bundle(analysis_id)

    return {
        "saved": True,
        "analysis_id": analysis_id,
        "orig_image_path": orig_saved_path,
        "seg_enabled": seg_enabled,
        "seg_overlay_path": overlay_saved_path,
    }
