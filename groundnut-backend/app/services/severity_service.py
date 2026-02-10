# app/services/severity_service.py
import numpy as np
from app.core.config import Config
from app.ml.severity.leaf_mask.predict import predict_leaf_mask

# Horsfall & Barratt (1945) ordinal categories (commonly used)
# Range (%) and midpoint (%) are widely referenced in phytopathometry literature.
# Categories below are expressed as severity % ranges:
# 0, 0-3, 3-6, 6-12, 12-25, 25-50, 50-75, 75-88, 88-94, 94-97, 97-100, 100
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

# Midpoints often used to convert ordinal classes into ratio-scale estimates
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
    """
    Map severity percentage -> SAD class using Horsfall–Barratt categories.
    Returns dict with:
      - class_index: 0..11  (ordinal class)
      - range_pct: [low, high]
      - midpoint_pct: midpoint of that class
    """
    p = float(np.clip(severity_pct, 0.0, 100.0))

    # Find first range where p <= high (ranges are ordered)
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
    """
    seg_batch: (1,H,W,3) float [0..1]  (input segmentation)
    infected_mask_bin: (H,W) uint8 {0,1} atau {0,255}

    severity = area_infected / area_leaf * 100%
    infected dihitung hanya pada area daun (leaf mask).
    """
    if seg_batch is None:
        raise ValueError("seg_batch is None")
    if infected_mask_bin is None:
        raise ValueError("infected_mask_bin is None")

    # threshold leaf mask (default dari Config, kamu minta 0.5)
    if thr is None:
        thr = float(getattr(Config, "SEV_LEAF_MASK_THRESHOLD", 0.5))
    thr = float(thr)

    # leaf mask dari model severity
    # predict_leaf_mask() mengembalikan: (prob_mask, leaf_mask_bin)
    prob_mask, leaf_mask_bin = predict_leaf_mask(seg_batch)

    # binarisasi leaf mask:
    # - kalau leaf_mask_bin sudah biner 0/1 atau 0/255: aman
    # - kalau yang kamu mau sebenarnya threshold prob_mask: gunakan prob_mask
    # pilih yang paling robust: kalau prob_mask ada, threshold di prob_mask
    leaf_bin = None
    if prob_mask is not None:
        leaf_bin = (np.array(prob_mask) >= thr).astype(np.uint8)
    else:
        leaf_arr = np.array(leaf_mask_bin)
        leaf_bin = (leaf_arr > 0).astype(np.uint8)

    infected = (np.array(infected_mask_bin) > 0).astype(np.uint8)

    leaf_area = int(leaf_bin.sum())
    infected_area = int((infected * leaf_bin).sum())  # hanya area daun

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
        "leaf_mask_bin": leaf_bin,  # (H,W) uint8 0/1
        "threshold": float(thr),
    }
