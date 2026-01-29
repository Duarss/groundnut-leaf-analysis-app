# app/ml/classification/predict.py
import json
import numpy as np
from app.core.config import Config
from .model_loader import get_classification_model

_idx_to_class_cache = None

def _load_idx_to_class():
    """
    Membaca classification_class_indices.json dan mengembalikan mapping:
      class_name -> idx, dan idx -> class_name
    """
    with open(Config.CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return class_indices, idx_to_class

def predict_leaf_class(preprocessed_batch):
    """
    preprocessed_batch:
        numpy array shape (1, H, W, 3) yang sudah di-preprocess_input.

    return:
        predicted_label: str
        confidence: float
        probs_dict: dict {label: prob}
    """
    global _idx_to_class_cache
    
    model = get_classification_model()

    if _idx_to_class_cache is None:
        _, _idx_to_class_cache = _load_idx_to_class()

    preds = model.predict(preprocessed_batch)
    probs = preds[0]

    # Index prediksi tertinggi
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    predicted_label = _idx_to_class_cache.get(idx, f"Class-{idx}")

    # ===== SORT PROBABILITIES (DESCENDING) =====
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

