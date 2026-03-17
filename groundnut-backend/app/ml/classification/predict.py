# app/ml/classification/predict.py
import json
import numpy as np
from app.core.config import Config
from .model_loader import get_classification_model

_idx_to_class_cache = None

def _load_idx_to_class():
    with open(Config.CLSF_CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return class_indices, idx_to_class

def predict_leaf_class(preprocessed_batch):
    global _idx_to_class_cache
    
    model = get_classification_model()

    if _idx_to_class_cache is None:
        _, _idx_to_class_cache = _load_idx_to_class()

    preds = model.predict(preprocessed_batch)
    probs = preds[0]

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    predicted_label = _idx_to_class_cache.get(idx, f"Class-{idx}")

    sorted_items = sorted(
        enumerate(probs),
        key=lambda x: x[1],
        reverse=True
    )

    probs_dict = {
        _idx_to_class_cache.get(i, f"Class-{i}"): float(p)
        for i, p in sorted_items
    }

    return predicted_label, confidence, probs_dict
