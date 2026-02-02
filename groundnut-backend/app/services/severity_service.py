# app/services/severity_service.py
import numpy as np
from app.core.config import Config
from app.ml.severity.leaf_mask.predict import predict_leaf_mask


def _parse_bins(s: str):
    """
    Expect string "0,5,20,40,60,100"
    """
    try:
        parts = [float(x.strip()) for x in str(s).split(",")]
        if len(parts) != 6:
            raise ValueError
        return parts  # [0,5,20,40,60,100]
    except Exception:
        return [0.0, 5.0, 20.0, 40.0, 60.0, 100.0]


def map_fao_level(severity_pct: float):
    """
    5 level:
      L1: 0-5
      L2: >5-20
      L3: >20-40
      L4: >40-60
      L5: >60-100
    """
    b = _parse_bins(getattr(Config, "SEV_FAO_BINS", "0,5,20,40,60,100"))
    p = float(np.clip(severity_pct, 0.0, 100.0))

    if p <= b[1]:
        lvl = 1
        rng = (b[0], b[1])
    elif p <= b[2]:
        lvl = 2
        rng = (b[1], b[2])
    elif p <= b[3]:
        lvl = 3
        rng = (b[2], b[3])
    elif p <= b[4]:
        lvl = 4
        rng = (b[3], b[4])
    else:
        lvl = 5
        rng = (b[4], b[5])

    return {"level": int(lvl), "range_pct": [float(rng[0]), float(rng[1])]}


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

    # threshold leaf mask (disepakati 0.5)
    if thr is None:
        thr = float(getattr(Config, "SEV_LEAF_MASK_THRESHOLD", 0.5))
    thr = float(thr)

    # leaf mask dari model severity
    # predict_leaf_mask() di project kamu mengembalikan: (prob_mask, leaf_mask_bin)
    _, leaf_mask_bin = predict_leaf_mask(seg_batch)  # (H,W) biasanya 0/1 atau 0/255

    infected = (np.array(infected_mask_bin) > 0).astype(np.uint8)

    leaf_arr = np.array(leaf_mask_bin)
    # kalau leaf_mask_bin keluaran 0/255 atau 0/1, ini aman
    leaf = (leaf_arr > 0).astype(np.uint8)

    leaf_area = int(leaf.sum())
    infected_area = int((infected * leaf).sum())  # hanya area daun

    if leaf_area <= 0:
        severity_pct = 0.0
    else:
        severity_pct = float(infected_area / float(leaf_area) * 100.0)

    fao = map_fao_level(severity_pct)

    # ✅ return 1 dict (lebih enak untuk service & frontend)
    # leaf_mask_bin kita ikutkan agar segmentation_service bisa:
    # - simpan PNG ke tmp_uploads
    # - dan buat base64 untuk tombol "Lihat mask daun"
    return {
        "severity_pct": float(severity_pct),
        "leaf_area_px": int(leaf_area),
        "infected_area_px": int(infected_area),
        "fao": fao,
        "leaf_mask_bin": leaf,  # (H,W) uint8 0/1
        "threshold": float(thr),
    }
