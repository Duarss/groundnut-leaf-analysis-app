# app/ml/segmentation/predict.py
import base64
import io
import numpy as np
from PIL import Image

from app.core.config import Config
from .model_loader import (
    get_segmentation_model, CLASS_TO_INDEX
)

HEALTHY_ALIASES = {"healthy", "HEALTHY", "Healthy"}

def _png_base64_rgb(arr_uint8_rgb: np.ndarray) -> str:
    img = Image.fromarray(arr_uint8_rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _mask_to_rgba(mask01, color=(255, 0, 0), alpha=110):
    """
    mask01: (H,W) float/uint 0/1
    menghasilkan RGBA overlay solid color dengan alpha mengikuti mask
    """
    m = (mask01 > 0.5).astype(np.uint8) * int(alpha)
    H, W = m.shape[:2]
    r = np.full((H, W), color[0], np.uint8)
    g = np.full((H, W), color[1], np.uint8)
    b = np.full((H, W), color[2], np.uint8)
    a = m.astype(np.uint8)
    return np.stack([r, g, b, a], axis=-1)


def predict_infected_areas(
    seg_batch_01,
    label: str,
    thr=None,
    overlay_alpha=None,
    return_mask: bool = False,
):
    """
    seg_batch_01: numpy (1,H,W,3) range [0..1]
    label: harus salah satu dari 4 kelas segmentasi (uppercase lebih aman)
    return_mask:
        - False (default): hanya return overlay + metadata
        - True: tambahkan infected_mask_bin (uint8 0/1) untuk dipakai severity
    """
    if str(label) in HEALTHY_ALIASES:
        return {"enabled": False, "reason": "Healthy -> tidak lanjut segmentasi"}

    if label not in CLASS_TO_INDEX:
        return {"enabled": False, "reason": f"Label '{label}' bukan 4 kelas segmentasi"}

    model = get_segmentation_model()
    pred = model.predict(seg_batch_01, verbose=0)[0]  # (H,W,4)

    ci = int(CLASS_TO_INDEX[label])
    prob = pred[..., ci]  # (H,W)

    thr = float(Config.SEG_MASK_THRESHOLD if thr is None else thr)
    bin_mask01 = (prob >= thr).astype(np.float32)   # (H,W) float 0/1
    bin_mask_u8 = (bin_mask01 > 0.5).astype(np.uint8)  # (H,W) uint8 0/1

    # base image (RGB) dari input 0..1
    base_rgb = np.clip(seg_batch_01[0] * 255.0, 0, 255).astype(np.uint8)
    base_rgba = Image.fromarray(base_rgb, mode="RGB").convert("RGBA")

    alpha = float(Config.SEG_OVERLAY_ALPHA if overlay_alpha is None else overlay_alpha)
    alpha = max(0.0, min(alpha, 1.0))

    # overlay alpha dikontrol via channel A mask RGBA
    mask_rgba = _mask_to_rgba(bin_mask01, color=(255, 0, 0), alpha=int(255 * alpha))
    pr_rgba = Image.fromarray(mask_rgba, mode="RGBA")

    over = Image.alpha_composite(base_rgba, pr_rgba).convert("RGB")
    overlay_b64 = _png_base64_rgb(np.array(over, dtype=np.uint8))

    out = {
        "enabled": True,
        "label": label,
        "channel_index": ci,
        "threshold": thr,
        "overlay_png_base64": overlay_b64,
    }

    # untuk severity estimation
    if return_mask:
        out["infected_mask_bin"] = bin_mask_u8  # numpy uint8 0/1 (akan dipop di service sebelum ditulis ke meta)

    return out
