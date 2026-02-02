# app/services/segmentation_service.py
import base64
import io
import os

import numpy as np
from PIL import Image

from app.core.config import Config
from app.utils.temp_store import (
    read_meta, read_image_bytes, write_meta, is_expired, delete_bundle
)
from app.utils.image_io import load_image_for_segmentation
from app.ml.segmentation.predict import predict_infected_areas
from app.services.severity_service import estimate_severity


# ==========================
# Toggle (aman untuk VRAM 4GB)
# ==========================
# infected mask base64 bisa lumayan besar (480x640 png), jadi default False.
SEND_INFECTED_MASK_B64 = False


def _write_png_bytes(path: str, png_bytes: bytes) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(png_bytes)
    return path


def _png_bytes_from_mask01(mask01_uint8: np.ndarray) -> bytes:
    """
    mask01_uint8: (H,W) uint8 {0,1} atau {0,255}
    return PNG bytes grayscale L (0/255)
    """
    m = np.asarray(mask01_uint8).astype(np.uint8)
    if m.size > 0 and m.max() <= 1:
        m = m * 255
    img = Image.fromarray(m, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mask01_to_b64_png(mask01_uint8: np.ndarray) -> str:
    png_bytes = _png_bytes_from_mask01(mask01_uint8)
    return base64.b64encode(png_bytes).decode("utf-8")


def _write_overlay_png_to_tmp(analysis_id: str, overlay_b64: str) -> str:
    """
    Simpan overlay PNG (base64) ke tmp_uploads agar bisa di-persist saat klik Save.
    """
    if not overlay_b64:
        raise ValueError("overlay_png_base64 kosong.")

    os.makedirs(Config.TEMP_DIR, exist_ok=True)
    out_path = os.path.join(Config.TEMP_DIR, f"{analysis_id}_overlay.png")
    _write_png_bytes(out_path, base64.b64decode(overlay_b64))
    return out_path


def segment_infected_areas(analysis_id: str):
    meta = read_meta(analysis_id)

    # meta kadaluarsa
    if is_expired(meta):
        delete_bundle(analysis_id)
        raise ValueError("Data kadaluarsa. Silakan upload ulang.")

    # label klasifikasi
    label = (meta.get("classification") or {}).get("label")
    if not label:
        raise ValueError("Label klasifikasi tidak ditemukan.")

    label_norm = str(label).strip().upper()

    # load image (reuse dari tmp_uploads)
    img_bytes = read_image_bytes(analysis_id)
    seg_batch = load_image_for_segmentation(img_bytes)

    # === SEGMENTATION (harus bisa return infected mask untuk severity) ===
    # NOTE: predict_infected_areas() harus sudah kamu update agar support return_mask=True
    seg_result = predict_infected_areas(seg_batch, label_norm, return_mask=True)

    meta["stage"] = meta.get("stage", "classified")

    # kalau segmentasi disabled, simpan & return
    if not seg_result.get("enabled"):
        meta["segmentation"] = seg_result
        write_meta(analysis_id, meta)
        return seg_result

    # === Overlay PNG ===
    overlay_b64 = seg_result.get("overlay_png_base64")
    if overlay_b64:
        overlay_path = _write_overlay_png_to_tmp(analysis_id, overlay_b64)
        seg_result["overlay_path"] = overlay_path

    # === Infected mask bin (wajib untuk severity) ===
    infected_mask_bin = seg_result.get("infected_mask_bin", None)
    if infected_mask_bin is None:
        # ini berarti predict_infected_areas belum benar-benar mengirim mask
        raise ValueError("infected_mask_bin tidak ditemukan dari hasil segmentasi. Pastikan return_mask=True didukung.")

    # simpan infected mask ke tmp_uploads
    infected_mask_path = os.path.join(Config.TEMP_DIR, f"{analysis_id}_infected_mask.png")
    infected_png = _png_bytes_from_mask01(infected_mask_bin)
    _write_png_bytes(infected_mask_path, infected_png)
    seg_result["infected_mask_path"] = infected_mask_path

    if SEND_INFECTED_MASK_B64:
        seg_result["infected_mask_png_base64"] = base64.b64encode(infected_png).decode("utf-8")

    # jangan simpan array besar ke meta JSON
    seg_result.pop("infected_mask_bin", None)

    # === SEVERITY ESTIMATION ===
    # severity_service.py kamu return dict (bukan tuple)
    severity_out = estimate_severity(seg_batch, infected_mask_bin)

    if not isinstance(severity_out, dict):
        raise ValueError("estimate_severity() harus mengembalikan dict.")

    # leaf mask ada di dict -> kita simpan PNG + base64 untuk tombol "Lihat mask daun"
    leaf_mask_bin = severity_out.get("leaf_mask_bin", None)
    if leaf_mask_bin is not None:
        leaf_mask_path = os.path.join(Config.TEMP_DIR, f"{analysis_id}_leaf_mask.png")
        leaf_png = _png_bytes_from_mask01(leaf_mask_bin)
        _write_png_bytes(leaf_mask_path, leaf_png)

        severity_out["leaf_mask_path"] = leaf_mask_path
        # key yang kamu mau cek di frontend:
        severity_out["leaf_mask_png_base64"] = base64.b64encode(leaf_png).decode("utf-8")

        # buang array besar biar meta.json ringan
        severity_out.pop("leaf_mask_bin", None)

    seg_result["severity"] = severity_out

    # simpan meta
    meta["stage"] = "segmented"
    meta["segmentation"] = seg_result
    write_meta(analysis_id, meta)

    return seg_result
