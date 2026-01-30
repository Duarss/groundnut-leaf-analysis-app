# app/ml/segmentation/predict.py
import base64
from cProfile import label
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
    m = (mask01 > 0.5).astype(np.uint8) * int(alpha)
    H, W = m.shape[:2]
    r = np.full((H, W), color[0], np.uint8)
    g = np.full((H, W), color[1], np.uint8)
    b = np.full((H, W), color[2], np.uint8)
    a = m.astype(np.uint8)
    return np.stack([r, g, b, a], axis=-1)


def predict_infected_areas(seg_batch_01, label: str, thr=None, overlay_alpha=None):
    if str(label) in HEALTHY_ALIASES:
        return {"enabled": False, "reason": "Healthy -> tidak lanjut segmentasi"}

    if label not in CLASS_TO_INDEX:
        return {"enabled": False, "reason": f"Label '{label}' bukan 4 kelas segmentasi"}

    model = get_segmentation_model()
    pred = model.predict(seg_batch_01, verbose=0)[0]  # (H,W,4)

    ci = int(CLASS_TO_INDEX[label])
    prob = pred[..., ci]
    thr = float(Config.SEG_MASK_THRESHOLD if thr is None else thr)

    bin_mask = (prob >= thr).astype(np.float32)

    # base image (RGB) dari input 0..1
    base_rgb = np.clip(seg_batch_01[0] * 255.0, 0, 255).astype(np.uint8)
    base_rgba = Image.fromarray(base_rgb, mode="RGB").convert("RGBA")

    alpha = float(Config.SEG_OVERLAY_ALPHA if overlay_alpha is None else overlay_alpha)
    # overlay alpha dikontrol via channel A mask RGBA (mirip exe_segmentation_model.py)
    mask_rgba = _mask_to_rgba(bin_mask, color=(255, 0, 0), alpha=int(255 * max(0.0, min(alpha, 1.0))))
    pr_rgba = Image.fromarray(mask_rgba, mode="RGBA")

    over = Image.alpha_composite(base_rgba, pr_rgba).convert("RGB")
    overlay_b64 = _png_base64_rgb(np.array(over, dtype=np.uint8))

    return {
        "enabled": True,
        "label": label,
        "channel_index": ci,
        "threshold": thr,
        "overlay_png_base64": overlay_b64,
    }
