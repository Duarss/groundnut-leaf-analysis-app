# app/ml/classification/model_loader.py
import os
import json

from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate,
    Dropout, Dense
)
from tensorflow.keras.regularizers import l2
from app.core.config import Config

_model = None

def _load_best_tuned_cfg():
    path = Config.BEST_CLSF_TUNED_CFG_PATH
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"best_tuned_cfg.json not found. Expected at: {path}. "
            "Please copy models/classification/best_tuned_cfg.json from training output."
        )
    with open(path, "r") as f:
        cfg = json.load(f)

    # minimal validation
    if "head" not in cfg or "head_type" not in cfg["head"]:
        raise ValueError("best_tuned_cfg.json invalid: missing cfg['head']['head_type']")
    return cfg


def _apply_head(x, head_cfg, num_classes):
    head_type = str(head_cfg.get("head_type"))

    drop1 = float(head_cfg.get("drop1"))
    drop2 = float(head_cfg.get("drop2"))
    dense_units = int(head_cfg.get("dense_units"))
    dense_l2 = float(head_cfg.get("dense_l2", 0.0))
    reg = l2(dense_l2) if dense_l2 > 0 else None

    if head_type == "gap_gmp":
        gap = GlobalAveragePooling2D(name="gap")(x)
        gmp = GlobalMaxPooling2D(name="gmp")(x)
        x = Concatenate(name="gap_gmp_concat")([gap, gmp])
    else:
        x = GlobalAveragePooling2D(name="gap")(x)

    act = "relu"
    if head_type in ("swish", "mlp2"):
        act = "swish"

    x = Dropout(drop1, name="drop1")(x)
    x = Dense(dense_units, activation=act, kernel_regularizer=reg, name="dense")(x)
    x = Dropout(drop2, name="drop2")(x)

    if head_type == "mlp2":
        dense_units2 = int(head_cfg.get("dense_units2", max(64, dense_units // 2)))
        drop3 = float(head_cfg.get("drop3", 0.2))
        x = Dense(dense_units2, activation=act, kernel_regularizer=reg, name="dense2")(x)
        x = Dropout(drop3, name="drop3")(x)

    out = Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)
    return out


def build_classification_model():
    img_size = (Config.CLSF_IMG_H, Config.CLSF_IMG_W)
    cfg = _load_best_tuned_cfg()

    base = EfficientNetB4(
        weights=None,
        include_top=False,
        input_shape=(img_size[0], img_size[1], 3),
    )
    out = _apply_head(base.output, cfg["head"], Config.CLSF_NUM_CLASSES)
    model = Model(inputs=base.input, outputs=out, name="EffNetB4_Classifier")
    return model


def get_classification_model():
    global _model
    if _model is None:
        model = build_classification_model()
        model.load_weights(Config.BEST_CLSF_WEIGHTS_PATH)
        _model = model
    return _model
