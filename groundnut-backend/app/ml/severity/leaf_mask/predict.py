# app/ml/severity/predict.py
import numpy as np
from app.core.config import Config
from app.ml.severity.leaf_mask.model_loader import get_severity_model


def predict_leaf_mask(seg_preprocessed_batch):
    """
    Input:
      seg_preprocessed_batch: numpy (1, H, W, 3) float [0..1]
      (pakai output dari load_image_for_segmentation agar konsisten)
    Output:
      mask_prob: (H, W) float32 [0..1]
      mask_bin:  (H, W) uint8 {0,1}
    """
    model = get_severity_model()
    pred = model.predict(seg_preprocessed_batch, verbose=0)  # (1,H,W,1)
    prob = pred[0, :, :, 0].astype(np.float32)

    thr = float(Config.SEV_LEAF_MASK_THRESHOLD)
    mask_bin = (prob >= thr).astype(np.uint8)

    return prob, mask_bin
