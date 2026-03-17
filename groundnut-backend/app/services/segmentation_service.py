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
from app.utils.image_io import load_image_for_segmentation, rotate_back_if_needed_pil
from app.ml.segmentation.predict import predict_infected_areas
from app.services.severity_service import estimate_severity

# ==========================
# Toggle (aman untuk VRAM 4GB)
# ==========================
# infected mask base64 bisa besar (480x640 PNG), jadi default False.
SEND_INFECTED_MASK_B64 = False

def _write_png_bytes(path: str, png_bytes: bytes) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(png_bytes)
    return path

def _png_bytes_from_mask01(mask01_uint8: np.ndarray) -> bytes:
    m = np.asarray(mask01_uint8).astype(np.uint8)
    if m.size > 0 and m.max() <= 1:
        m = m * 255
    img = Image.fromarray(m, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def _decode_b64_png_to_pil(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")

def _encode_pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _rotate_mask_back(mask_hw: np.ndarray, rotated: bool) -> np.ndarray:
    if not rotated:
        return mask_hw
    return np.rot90(mask_hw, k=3)

def _write_overlay_png_to_tmp(analysis_id: str, overlay_b64: str) -> str:
    if not overlay_b64:
        raise ValueError("overlay_png_base64 kosong.")
    os.makedirs(Config.TEMP_DIR, exist_ok=True)
    out_path = os.path.join(Config.TEMP_DIR, f"{analysis_id}_overlay.png")
    _write_png_bytes(out_path, base64.b64decode(overlay_b64))
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
        raise ValueError("Label klasifikasi tidak ditemukan. Silahkan lakukan klasifikasi ulang.")

    label_norm = str(label).strip().upper()
    img_bytes = read_image_bytes(analysis_id)
    seg_batch, rotated_flag = load_image_for_segmentation(img_bytes, return_rotated=True)

    seg_result = predict_infected_areas(seg_batch, label_norm, return_mask=True)

    meta["stage"] = meta.get("stage", "classified")

    if seg_result.get("enabled"):
        overlay_b64 = seg_result.get("overlay_png_base64")
        if overlay_b64:
            try:
                over_img = _decode_b64_png_to_pil(overlay_b64)
                over_img = rotate_back_if_needed_pil(over_img, rotated_flag)
                overlay_b64 = _encode_pil_to_b64_png(over_img)
                seg_result["overlay_png_base64"] = overlay_b64
            except Exception:
                pass

            overlay_path = _write_overlay_png_to_tmp(analysis_id, overlay_b64)
            seg_result["overlay_path"] = overlay_path

        infected_mask_bin = seg_result.get("infected_mask_bin", None)
        if infected_mask_bin is None:
            raise RuntimeError("predict_infected_areas() tidak mengembalikan infected_mask_bin.")
        
        infected_mask_bin = np.asarray(infected_mask_bin).astype(np.uint8)
        infected_mask_path = os.path.join(Config.TEMP_DIR, f"{analysis_id}_infected_mask.png")
        infected_mask_for_calc = infected_mask_bin
        infected_mask_to_save = _rotate_mask_back(infected_mask_bin, rotated_flag)

        png_bytes = _png_bytes_from_mask01(infected_mask_to_save)
        _write_png_bytes(infected_mask_path, png_bytes)

        seg_result["infected_mask_path"] = infected_mask_path

        if SEND_INFECTED_MASK_B64:
            seg_result["infected_mask_png_base64"] = base64.b64encode(png_bytes).decode("utf-8")

        seg_result.pop("infected_mask_bin", None)

        severity_out = estimate_severity(seg_batch, infected_mask_bin=infected_mask_for_calc)

        if isinstance(severity_out, dict):
            leaf_mask_bin = severity_out.get("leaf_mask_bin")

            if leaf_mask_bin is not None:
                leaf_mask_arr = np.asarray(leaf_mask_bin, dtype=np.uint8)

                leaf_mask_arr_to_save = _rotate_mask_back(leaf_mask_arr, rotated_flag)
                leaf_mask_path = os.path.join(Config.TEMP_DIR, f"{analysis_id}_leaf_mask.png")
                png_leaf = _png_bytes_from_mask01(leaf_mask_arr_to_save)

                _write_png_bytes(leaf_mask_path, png_leaf)

                severity_out["leaf_mask_path"] = leaf_mask_path
                severity_out["leaf_mask_png_base64"] = base64.b64encode(png_leaf).decode("utf-8")

                severity_out.pop("leaf_mask_bin", None)

            for k in list(severity_out.keys()):
                v = severity_out.get(k)
                if isinstance(v, np.ndarray):
                    severity_out.pop(k, None)

            seg_result["severity"] = severity_out

        meta["stage"] = "segmented"

    meta["segmentation"] = seg_result
    write_meta(analysis_id, meta)
    return seg_result
