# app/ml/severity/leaf_mask/model_loader.py
import json
import os
import threading
import tensorflow as tf
from tensorflow.keras import mixed_precision
from app.core.config import Config

# ikuti policy project (mixed float16 oke untuk inferensi; output nanti float32)
mixed_precision.set_global_policy("mixed_float16")

_model = None
_lock = threading.Lock()

def _load_cfg():
    path = Config.BEST_SEV_CFG_PATH
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"best_tuned_cfg.json tidak ditemukan di: {path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def conv_block(x, f, use_bn=True, dropout=0.0, sep_conv=False):
    Conv = tf.keras.layers.SeparableConv2D if sep_conv else tf.keras.layers.Conv2D

    x = Conv(f, 3, padding="same")(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = Conv(f, 3, padding="same")(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def build_unet(cfg, input_shape=(480, 640, 3)):
    base_filters = int(cfg.get("base_filters", 12))
    depth = int(cfg.get("depth", 3))
    use_bn = bool(cfg.get("use_bn", True))
    dropout = float(cfg.get("dropout", 0.0))
    sep_conv = bool(cfg.get("sep_conv", False))

    inp = tf.keras.Input(shape=input_shape)
    skips = []

    x = inp
    f = base_filters
    for d in range(depth):
        x = conv_block(x, f, use_bn=use_bn, dropout=dropout, sep_conv=sep_conv)
        skips.append(x)
        x = tf.keras.layers.MaxPool2D()(x)
        f *= 2

    x = conv_block(x, f, use_bn=use_bn, dropout=dropout, sep_conv=sep_conv)

    for d in reversed(range(depth)):
        f //= 2
        x = tf.keras.layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = tf.keras.layers.Concatenate()([x, skips[d]])
        x = conv_block(x, f, use_bn=use_bn, dropout=dropout, sep_conv=sep_conv)

    out = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", use_bias=True, dtype="float32", name="leaf_mask")(x)
    return tf.keras.Model(inp, out, name="SeverityLeafMaskUNet")


def get_severity_model():
    global _model
    if _model is not None:
        return _model

    with _lock:
        if _model is not None:
            return _model

        cfg = _load_cfg()
        net_h, net_w = int(Config.SEG_IMG_H), int(Config.SEG_IMG_W)
        m = build_unet(cfg, input_shape=(net_h, net_w, 3))
        if isinstance(m, (tuple, list)):
            m = m[0]
        m.load_weights(Config.BEST_SEV_WEIGHTS_PATH)
        _model = m

        return _model
