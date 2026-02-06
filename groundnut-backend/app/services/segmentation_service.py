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
SEND_INFECTED_MASK_B64 = False  # default False (png mask bisa besar)


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


def _decode_b64_png_to_pil(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _encode_pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _rotate_mask_back(mask_hw: np.ndarray, rotated: bool) -> np.ndarray:
    """
    Jika input portrait kita ROTATE_90 (CCW) sebelum model,
    maka output mask perlu ROTATE_270 (CCW) agar kembali ke orientasi semula.
    Numpy: rot90(k=3) = 270 CCW.
    """
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

    if is_expired(meta):
        delete_bundle(analysis_id)
        raise ValueError("Data kadaluarsa. Silakan upload ulang.")

    label = (meta.get("classification") or {}).get("label")
    if not label:
        raise ValueError("Label klasifikasi tidak ditemukan.")

    label_norm = str(label).strip().upper()

    img_bytes = read_image_bytes(analysis_id)

    # IMPORTANT: untuk segmentasi, kita izinkan rotate internal jika portrait
    seg_batch, rotated_flag = load_image_for_segmentation(img_bytes, return_rotated=True)

    # === SEGMENTATION (minta infected_mask_bin untuk severity) ===
    seg_result = predict_infected_areas(
        seg_batch,
        label_norm,
        return_mask=True,
    )

    meta["stage"] = meta.get("stage", "classified")

    # kalau disabled, simpan & return apa adanya
    if not seg_result.get("enabled"):
        meta["segmentation"] = seg_result
        write_meta(analysis_id, meta)
        return seg_result

    # ==========================
    # 1) Overlay: rotate back jika portrait awal
    # ==========================
    overlay_b64 = seg_result.get("overlay_png_base64")
    if overlay_b64:
        try:
            over_img = _decode_b64_png_to_pil(overlay_b64)
            over_img = rotate_back_if_needed_pil(over_img, rotated_flag)
            overlay_b64 = _encode_pil_to_b64_png(over_img)
            seg_result["overlay_png_base64"] = overlay_b64
        except Exception:
            # kalau gagal decode/encode, biarkan overlay apa adanya
            pass

        overlay_path = _write_overlay_png_to_tmp(analysis_id, overlay_b64)
        seg_result["overlay_path"] = overlay_path

    # ==========================
    # 2) Infected mask: simpan sementara
    #    - untuk severity: pakai versi ROTATED (match seg_batch)
    #    - untuk simpan/tampil: rotate back supaya sama dengan orientasi user
    # ==========================
    infected_mask_bin = seg_result.get("infected_mask_bin", None)
    if infected_mask_bin is None:
        raise RuntimeError("predict_infected_areas() tidak mengembalikan infected_mask_bin padahal return_mask=True.")

    infected_mask_bin = np.asarray(infected_mask_bin).astype(np.uint8)

    infected_mask_path = os.path.join(Config.TEMP_DIR, f"{analysis_id}_infected_mask.png")
    infected_mask_to_save = _rotate_mask_back(infected_mask_bin, rotated_flag)
    png_inf = _png_bytes_from_mask01(infected_mask_to_save)
    _write_png_bytes(infected_mask_path, png_inf)
    seg_result["infected_mask_path"] = infected_mask_path

    if SEND_INFECTED_MASK_B64:
        seg_result["infected_mask_png_base64"] = base64.b64encode(png_inf).decode("utf-8")

    # jangan simpan array besar di response/meta
    seg_result.pop("infected_mask_bin", None)

    # ==========================
    # 3) SEVERITY ESTIMATION
    # ==========================
    # NOTE: estimate_severity() kamu return dict (bukan tuple)
    severity_out = estimate_severity(
        seg_batch,
        infected_mask_bin,  # versi rotated (matching seg_batch)
    )

    if not isinstance(severity_out, dict):
        raise RuntimeError("estimate_severity() harus mengembalikan dict.")

    # ambil leaf_mask_bin dari dict (untuk tombol 'lihat mask daun')
    leaf_mask_bin = severity_out.get("leaf_mask_bin", None)
    if leaf_mask_bin is None:
        # masih bisa jalan tanpa leaf mask, tapi tombol mask daun tidak ada
        # (kamu bisa pilih raise kalau wajib)
        seg_result["severity"] = severity_out
    else:
        leaf_mask_bin = np.asarray(leaf_mask_bin).astype(np.uint8)
        leaf_mask_to_save = _rotate_mask_back(leaf_mask_bin, rotated_flag)

        leaf_mask_path = os.path.join(Config.TEMP_DIR, f"{analysis_id}_leaf_mask.png")
        png_leaf = _png_bytes_from_mask01(leaf_mask_to_save)
        _write_png_bytes(leaf_mask_path, png_leaf)

        # simpan path + base64 untuk frontend
        severity_out["leaf_mask_path"] = leaf_mask_path
        severity_out["leaf_mask_png_base64"] = base64.b64encode(png_leaf).decode("utf-8")

        # jangan simpan array besar di meta/json response
        severity_out.pop("leaf_mask_bin", None)

        seg_result["severity"] = severity_out

    # update meta
    meta["stage"] = "segmented"
    meta["segmentation"] = seg_result
    write_meta(analysis_id, meta)

    return seg_result
