# app/ml/segmentation/model_loader.py
import json
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, Activation, Concatenate,
    SpatialDropout2D, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0, efficientnet

from app.core.config import Config

# ikuti exe_segmentation_model.py: output float32
mixed_precision.set_global_policy("mixed_float16")

CLASS_NAMES = [
    "ALTERNARIA LEAF SPOT",
    "LEAF SPOT (EARLY AND LATE)",
    "ROSETTE",
    "RUST",
]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASS_NAMES)}

# dropout globals (akan diset dari cfg)
BOT_DROPOUT = 0.0
DEC_DROPOUT = 0.0

_model = None

def _set_dropout_globals(cfg: dict):
    global BOT_DROPOUT, DEC_DROPOUT
    BOT_DROPOUT = float(cfg.get("bot_dropout", 0.0))
    DEC_DROPOUT = float(cfg.get("dec_dropout", 0.0))


def _conv(x, f, k=3, drop=0.0):
    x = Conv2D(f, k, padding="same", use_bias=True)(x)
    x = Activation("relu")(x)
    if drop and drop > 0:
        x = SpatialDropout2D(drop)(x)
    return x


def _safe_concat(x, skip, idx=0):
    def _resize_like(tensors):
        a, b = tensors
        h = tf.shape(a)[1]
        w = tf.shape(a)[2]
        return tf.image.resize(b, (h, w), method="bilinear")
    skip2 = Lambda(_resize_like, name=f"resize_skip_like_{idx}")([x, skip])
    return Concatenate(name=f"concat_{idx}")([x, skip2])


def _decoder_block(x, skip, f, drop, idx=0):
    x = Conv2DTranspose(f, 2, strides=2, padding="same", name=f"up_{idx}")(x)
    x = _safe_concat(x, skip, idx=idx)
    x = _conv(x, f, k=3, drop=drop)
    x = _conv(x, f, k=3, drop=0.0)
    return x


def build_unet_efficientnetb0(net_h, net_w, out_channels=4, train_encoder=False):
    inp = Input((net_h, net_w, 3))

    # sama seperti exe_segmentation_model.py:
    # input 0..1 -> scale 255 -> efficientnet.preprocess_input
    x0 = Lambda(lambda t: t * 255.0, name="scale_255")(inp)
    x0 = efficientnet.preprocess_input(x0)

    base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x0)
    base.trainable = bool(train_encoder)

    s1 = base.get_layer("stem_activation").output
    s2 = base.get_layer("block2a_activation").output
    s3 = base.get_layer("block3a_activation").output
    s4 = base.get_layer("block4a_activation").output
    b  = base.get_layer("top_activation").output

    b = _conv(b, 256, k=3, drop=BOT_DROPOUT)
    b = _conv(b, 256, k=3, drop=0.0)

    d4 = _decoder_block(b,  s4, 256, drop=DEC_DROPOUT, idx=4)
    d3 = _decoder_block(d4, s3, 128, drop=DEC_DROPOUT, idx=3)
    d2 = _decoder_block(d3, s2, 64,  drop=DEC_DROPOUT, idx=2)
    d1 = _decoder_block(d2, s1, 32,  drop=DEC_DROPOUT, idx=1)

    x = Conv2DTranspose(16, 2, strides=2, padding="same", name="up_final")(d1)
    x = _conv(x, 16, k=3, drop=0.0)

    out = Conv2D(out_channels, 1, activation="sigmoid", dtype="float32", name="mask")(x)
    model = Model(inp, out, name="UNet_EfficientNetB0")
    return model


def load_selected_cfg():
    with open(Config.BEST_SEG_TUNED_CFG_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_segmentation_model():
    """
    Load 1x model global4 dan cache di memory.
    """
    global _model
    if _model is not None:
        return _model

    cfg = load_selected_cfg()
    _set_dropout_globals(cfg)

    net_h, net_w = int(Config.SEG_IMG_H), int(Config.SEG_IMG_W)
    _model = build_unet_efficientnetb0(net_h, net_w, out_channels=4, train_encoder=False)
    _model.build((None, net_h, net_w, 3))
    _model.load_weights(Config.BEST_SEG_WEIGHTS_PATH)

    return _model
