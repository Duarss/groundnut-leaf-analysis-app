# app/services/severity_service.py
import numpy as np
from app.core.config import Config
from app.ml.severity.leaf_mask.predict import predict_leaf_mask

# Horsfall & Barratt (1945) ordinal categories (commonly used)
# Range (%) and midpoint (%) are widely referenced in phytopathometry literature.
HB_RANGES = [
    (0.0, 0.0),
    (0.0, 3.0),
    (3.0, 6.0),
    (6.0, 12.0),
    (12.0, 25.0),
    (25.0, 50.0),
    (50.0, 75.0),
    (75.0, 88.0),
    (88.0, 94.0),
    (94.0, 97.0),
    (97.0, 100.0),
    (100.0, 100.0),
]

HB_MIDPOINTS = [
    0.0,
    1.5,
    4.5,
    9.0,
    18.5,
    37.5,
    62.5,
    81.5,
    91.0,
    95.5,
    98.5,
    100.0,
]

def map_sad_class_hb(severity_pct: float):
    p = float(np.clip(severity_pct, 0.0, 100.0))

    idx = 0
    for i, (lo, hi) in enumerate(HB_RANGES):
        if p <= hi:
            idx = i
            break

    lo, hi = HB_RANGES[idx]
    mid = HB_MIDPOINTS[idx]
    return {
        "scheme": "Horsfall-Barratt (12-class)",
        "class_index": int(idx),
        "range_pct": [float(lo), float(hi)],
        "midpoint_pct": float(mid),
    }

def estimate_severity(seg_batch, infected_mask_bin, thr: float = None):
    if seg_batch is None:
        raise ValueError("seg_batch is None")
    if infected_mask_bin is None:
        raise ValueError("infected_mask_bin is None")

    if thr is None:
        thr = float(getattr(Config, "SEV_LEAF_MASK_THRESHOLD", 0.5))
    thr = float(thr)

    prob_mask, leaf_mask_bin = predict_leaf_mask(seg_batch)

    leaf_bin = None
    if prob_mask is not None:
        leaf_bin = (np.array(prob_mask) >= thr).astype(np.uint8)
    else:
        leaf_arr = np.array(leaf_mask_bin)
        leaf_bin = (leaf_arr > 0).astype(np.uint8)

    infected = (np.array(infected_mask_bin) > 0).astype(np.uint8)

    leaf_area = int(leaf_bin.sum())
    infected_area = int((infected * leaf_bin).sum())

    if leaf_area <= 0:
        severity_pct = 0.0
    else:
        severity_pct = float(infected_area / float(leaf_area) * 100.0)

    sad = map_sad_class_hb(severity_pct)

    return {
        "severity_pct": float(severity_pct),
        "leaf_area_px": int(leaf_area),
        "infected_area_px": int(infected_area),
        "sad": sad,
        "leaf_mask_bin": leaf_bin,
        "threshold": float(thr),
    }
