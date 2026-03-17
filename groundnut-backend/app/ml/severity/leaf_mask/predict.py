# app/ml/severity/predict.py
import numpy as np
from app.core.config import Config
from app.ml.severity.leaf_mask.model_loader import get_severity_model


def predict_leaf_mask(seg_preprocessed_batch):
    model = get_severity_model()
    pred = model.predict(seg_preprocessed_batch, verbose=0)
    prob = pred[0, :, :, 0].astype(np.float32)

    thr = float(Config.SEV_LEAF_MASK_THRESHOLD)
    mask_bin = (prob >= thr).astype(np.uint8)

    return prob, mask_bin
