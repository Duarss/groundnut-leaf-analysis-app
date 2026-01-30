# app/services/segmentation_service.py
import base64
import os

from app.core.config import Config
from app.utils.temp_store import (
    read_meta, read_image_bytes, write_meta, is_expired, delete_bundle
)
from app.utils.image_io import load_image_for_segmentation
from app.ml.segmentation.predict import predict_infected_areas

def _write_overlay_png_to_tmp(analysis_id: str, overlay_b64: str) -> str:
    """
    Simpan overlay PNG (base64) ke tmp_uploads agar bisa di-persist saar klik Save.
    """
    if not overlay_b64:
        return ValueError("overlay_png_base64 kosong.")
    
    os.makedirs(Config.TEMP_UPLOADS_DIR, exist_ok=True)
    out_path = os.path.join(Config.TEMP_UPLOADS_DIR, f"{analysis_id}_overlay.png")
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(overlay_b64))
    return out_path

def segment_infected_areas(analysis_id: str):
    meta = read_meta(analysis_id)

    if not meta:
        raise ValueError("Metadata tidak ditemukan. Silahkan lakukan klasifikasi ulang.")

    if is_expired(meta):
        delete_bundle(analysis_id)
        raise ValueError("Mohon upload gambar ulang: data sudah kadaluarsa.")

    cls = (meta.get("classification") or {})
    label = cls.get("label")

    if not label:
        # metadata ada, tapi label kosong → klasifikasi belum tersimpan/format beda
        raise ValueError("Label klasifikasi tidak ditemukan. Silahkan lakukan klasifikasi ulang.")

    # normalisasi agar cocok dengan CLASS_TO_INDEX (yang uppercase semua)
    label_norm = str(label).strip().upper()

    img_bytes = read_image_bytes(analysis_id)
    seg_batch = load_image_for_segmentation(img_bytes)

    seg_result = predict_infected_areas(seg_batch, label_norm)

    if seg_result.get("enabled"):
        overlay_b64 = seg_result.get("overlay_png_base64")
        if overlay_b64:
            overlay_path = _write_overlay_png_to_tmp(analysis_id, overlay_b64)
            seg_result["overlay_path"] = overlay_path

    # simpan ke meta kalau enabled
    meta["stage"] = "segmented" if seg_result.get("enabled") else meta.get("stage", "classified")
    meta["segmentation"] = seg_result
    write_meta(analysis_id, meta)

    return seg_result
