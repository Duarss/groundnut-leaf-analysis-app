# tools/segmentation_train_eval.py
import os, re, gc, csv, glob, json, random, argparse, itertools
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, Activation, Concatenate,
    SpatialDropout2D, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Metric
from tensorflow.keras.applications import EfficientNetB0, efficientnet

# =======================
# GLOBALS
# ======================
GLOBAL_SEED = 42

BASE_DIR  = "datasets/processed/segmentation_dataset"
MODEL_DIR = "models/segmentation"
RES_DIR   = "datasets/results/segmentation"

IMG_EXT  = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
MASK_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

CLASS_NAMES = [
    "ALTERNARIA LEAF SPOT",
    "LEAF SPOT (EARLY AND LATE)",
    "ROSETTE",
    "RUST",
]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASS_NAMES)}

BOT_DROPOUT = 0.0
DEC_DROPOUT = 0.0

# =======================
# SETUP / IO HELPERS
# =======================
def setup():
    mixed_precision.set_global_policy("mixed_float16")
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    except Exception:
        pass

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    tf.random.set_seed(GLOBAL_SEED)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

def cleanup(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass

def jsave(path: str, obj: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def jload(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_name(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "RUN"
    return "".join([c if c.isalnum() or c in ["_", "-"] else "_" for c in s])

def slug(s: str) -> str:
    return safe_name(s).lower()

def abbr_class_name(class_name: str) -> str:
    s = (class_name or "").strip().upper()
    tokens = re.findall(r"[A-Z]+", s)
    if not tokens:
        return "cls"
    if len(tokens) >= 2:
        return "".join(t[0] for t in tokens).lower()
    return tokens[0][:3].lower()

def short_tag(args) -> str:
    return "global"

def cfg_path(args) -> str:
    return os.path.join(MODEL_DIR, f"best_{short_tag(args)}_tuned_cfg.json")

def trials_csv_path(args) -> str:
    return os.path.join(MODEL_DIR, f"tune_{short_tag(args)}_{slug(args.search_method)}_trials.csv")

def artifact_paths(args):
    tag = short_tag(args)
    weights = os.path.join(MODEL_DIR, f"best_{tag}_segmentation_model.weights.h5")
    hist    = os.path.join(MODEL_DIR, f"best_{tag}_segmentation_model_hist.json")
    spec    = os.path.join(MODEL_DIR, f"best_{tag}_segmentation_model_spec.json")
    return weights, hist, spec, tag

def res_root(args) -> str:
    return os.path.join(RES_DIR, "global4")

# =======================
# CFG NORMALIZE / POLICY
# =======================
def normalize_cfg(cfg: dict) -> dict:
    c = dict(cfg)

    tv = float(np.clip(float(c.get("tv_w", 0.20)), 0.0, 1.0))
    c["tv_w"] = tv
    c["bce_w"] = float(1.0 - tv)
    s = float(c["tv_w"] + c["bce_w"])
    if s > 0:
        c["tv_w"] /= s
        c["bce_w"] /= s

    a = float(np.clip(float(c.get("tv_alpha", 0.30)), 0.0, 1.0))
    b = float(np.clip(float(c.get("tv_beta", 0.70)), 0.0, 1.0))
    ab = a + b
    if ab <= 0:
        a, b, ab = 0.5, 0.5, 1.0
    c["tv_alpha"] = float(a / ab)
    c["tv_beta"]  = float(b / ab)

    c["mix_full_ratio"] = float(c["mix_full_ratio"])
    c["lr"] = float(c["lr"])
    c["train_encoder"] = bool(int(c["train_encoder"])) if isinstance(c["train_encoder"], (int, np.integer)) else bool(c["train_encoder"])
    c["use_focal"]     = bool(int(c["use_focal"]))     if isinstance(c["use_focal"], (int, np.integer))     else bool(c["use_focal"])
    c["bot_dropout"] = float(c.get("bot_dropout", 0.0))
    c["dec_dropout"] = float(c.get("dec_dropout", 0.0))

    if "freeze_bn" in c:
        c["freeze_bn"] = bool(int(c["freeze_bn"])) if isinstance(c["freeze_bn"], (int, np.integer)) else bool(c["freeze_bn"])
    else:
        c["freeze_bn"] = True

    return c

def cfg_key(cfg: dict):
    keys = [
        "mix_full_ratio","lr","train_encoder","tv_w","bce_w",
        "tv_alpha","tv_beta","use_focal","bot_dropout","dec_dropout"
    ]
    return tuple(cfg.get(k) for k in keys)

def is_rosette_onlyclass(args) -> bool:
    return False

def apply_rosette_policy(args, cfg: dict) -> dict:
    # No special policies needed for global 4-class mode
    return cfg

def set_dropout_globals(cfg: dict):
    global BOT_DROPOUT, DEC_DROPOUT
    BOT_DROPOUT = float(cfg.get("bot_dropout", 0.0))
    DEC_DROPOUT = float(cfg.get("dec_dropout", 0.0))

def load_cfg(args) -> dict:
    if args.cfg_json:
        if not os.path.exists(args.cfg_json):
            raise RuntimeError(f"--cfg_json not found: {args.cfg_json}")
        raw = jload(args.cfg_json)
        cfg = raw.get("selected_cfg", raw)
        return normalize_cfg(cfg)

    p = cfg_path(args)
    if not os.path.exists(p):
        raise RuntimeError(
            "CFG tidak ditemukan.\n"
            f"- Expected: {p}\n"
            "- Jalankan dulu: --mode tune, atau supply: --cfg_json path.json"
        )
    raw = jload(p)
    cfg = raw.get("selected_cfg", raw)
    return normalize_cfg(cfg)

# =======================
# PLOTTING
# =======================
def plot_training_curves(history_dict, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    if not history_dict or not isinstance(history_dict, dict):
        print("[WARN] Empty/invalid history. Skip plotting.")
        return

    n_epochs = None
    for _, v in history_dict.items():
        if isinstance(v, (list, tuple)) and len(v) > 0:
            n_epochs = len(v)
            break
    if not n_epochs:
        print("[WARN] No epoch data. Skip plotting.")
        return

    epochs = np.arange(1, n_epochs + 1)

    def plot_pair(train_key, val_key, ylabel, title, filename, leg1, leg2):
        if train_key not in history_dict:
            print(f"[INFO] Missing {train_key}, skip {filename}")
            return
        plt.figure()
        plt.plot(epochs, history_dict.get(train_key, []), label=leg1)
        if val_key in history_dict:
            plt.plot(epochs, history_dict.get(val_key, []), label=leg2)
        plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title)
        plt.legend(); plt.tight_layout()
        out_path = os.path.join(out_dir, filename)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[OK] saved plot -> {out_path}")

    plot_pair("loss",   "val_loss",   "Loss",     "Training vs Validation Loss",      f"{prefix}_curve_loss.png", "train_loss", "val_loss")
    plot_pair("dice50", "val_dice50", "Dice@0.5", "Training vs Validation Dice@0.5",  f"{prefix}_curve_dice.png", "train_dice50", "val_dice50")
    plot_pair("iou50",  "val_iou50",  "IoU@0.5",  "Training vs Validation IoU@0.5",   f"{prefix}_curve_iou.png",  "train_iou50", "val_iou50")

# =======================
# DATA
# =======================
def _find_mask(mask_dir, stem):
    for ext in MASK_EXT:
        mp = os.path.join(mask_dir, stem + ext)
        if os.path.exists(mp):
            return mp
    return None

def _list_classes(root: str):
    if not os.path.isdir(root):
        return []
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def collect_pairs(root: str, args):
    pairs = []
    if not os.path.isdir(root):
        print(f"[WARN] split not found: {root}")
        return pairs

    classes = _list_classes(root)

    for cls in classes:
        cls_dir = os.path.join(root, cls)
        img_dir = os.path.join(cls_dir, "images")
        msk_dir = os.path.join(cls_dir, "masks")
        if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
            continue

        for ip in sorted(glob.glob(os.path.join(img_dir, "*"))):
            if not ip.lower().endswith(IMG_EXT):
                continue
            stem = Path(ip).stem
            mp = _find_mask(msk_dir, stem)
            if mp is not None:
                pairs.append((cls, ip, mp))

    msg = f"[DATA] {root}: {len(pairs)} pairs"
    print(msg)
    return pairs

def read_image(p, out_h: int, out_w: int):
    img = Image.open(p).convert("RGB")
    img = img.resize((out_w, out_h), Image.BILINEAR)
    arr = np.asarray(img, np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)

def read_mask_binary(p, out_h: int, out_w: int):
    m = Image.open(p).convert("L")
    m = m.resize((out_w, out_h), Image.NEAREST)
    arr = np.asarray(m, np.float32)
    arr = (arr > 127).astype(np.float32)
    return np.expand_dims(arr, -1)

def make_dataset(pairs, args, batch, shuffle=True, drop_remainder=False):
    out_h, out_w = int(args.img_h), int(args.img_w)
    n_classes = len(CLASS_NAMES)

    def gen():
        idx = np.arange(len(pairs))
        if shuffle:
            np.random.shuffle(idx)

        for i in idx:
            cls, ip, mp = pairs[i]
            x = read_image(ip, out_h, out_w)
            y_bin = read_mask_binary(mp, out_h, out_w)

            if cls not in CLASS_TO_INDEX:
                continue
            ci = CLASS_TO_INDEX[cls]
            y = np.zeros((out_h, out_w, n_classes), np.float32)
            y[..., ci] = y_bin[..., 0]

            yield x, y

    y_shape = (out_h, out_w, len(CLASS_NAMES))
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((out_h, out_w, 3), tf.float32),
            tf.TensorSpec(y_shape, tf.float32),
        )
    )
    if shuffle:
        ds = ds.shuffle(256, seed=GLOBAL_SEED, reshuffle_each_iteration=True)
    return ds.batch(batch, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE)

def build_train_dataset_mix(args, cfg, batch):
    r = float(np.clip(cfg["mix_full_ratio"], 0.0, 1.0))
    roi_root  = os.path.join(BASE_DIR, args.train_roi_split)
    full_root = os.path.join(BASE_DIR, args.train_full_split)

    if r <= 0.0:
        tr = collect_pairs(roi_root, args)
        if len(tr) == 0:
            raise RuntimeError("Train ROI pairs kosong. Cek train_roi_split.")
        ds = make_dataset(tr, args, batch, shuffle=True, drop_remainder=True)
        return ds, {"mode": "single_roi", "train_root": roi_root, "train_n": len(tr), "mix_full_ratio": r}

    if r >= 1.0:
        tr = collect_pairs(full_root, args)
        if len(tr) == 0:
            raise RuntimeError("Train FULL pairs kosong. Cek train_full_split.")
        ds = make_dataset(tr, args, batch, shuffle=True, drop_remainder=True)
        return ds, {"mode": "single_full", "train_root": full_root, "train_n": len(tr), "mix_full_ratio": r}

    tr_roi  = collect_pairs(roi_root, args)
    tr_full = collect_pairs(full_root, args)
    if len(tr_roi) == 0:
        raise RuntimeError("train_roi_split kosong / tidak ditemukan.")
    if len(tr_full) == 0:
        raise RuntimeError("train_full_split kosong / tidak ditemukan.")

    ds_roi  = make_dataset(tr_roi,  args, batch=1, shuffle=True, drop_remainder=True).unbatch()
    ds_full = make_dataset(tr_full, args, batch=1, shuffle=True, drop_remainder=True).unbatch()

    ds_mix = (
        tf.data.Dataset.sample_from_datasets([ds_roi, ds_full], weights=[1.0 - r, r], seed=GLOBAL_SEED)
        .batch(batch, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    info = {"mode": "mix", "roi_root": roi_root, "roi_n": len(tr_roi), "full_root": full_root, "full_n": len(tr_full), "mix_full_ratio": r}
    return ds_mix, info

# =======================
# METRICS
# =======================
class DiceBin(Metric):
    def __init__(self, thr=0.5, name=None, smooth=1.0, **kwargs):
        super().__init__(name=name or f"dice{int(thr*100):02d}", **kwargs)
        self.thr = float(thr)
        self.smooth = float(smooth)
        self.total = self.add_weight(name="total", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > self.thr, tf.float32)
        inter = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
        union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        self.total.assign_add(tf.reduce_mean(dice))
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

class IoUBin(Metric):
    def __init__(self, thr=0.5, name=None, smooth=1.0, **kwargs):
        super().__init__(name=name or f"iou{int(thr*100):02d}", **kwargs)
        self.thr = float(thr)
        self.smooth = float(smooth)
        self.total = self.add_weight(name="total", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > self.thr, tf.float32)
        inter = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
        union = tf.reduce_sum(y_true + y_pred - y_true*y_pred, axis=[1,2,3])
        iou = (inter + self.smooth) / (union + self.smooth)
        self.total.assign_add(tf.reduce_mean(iou))
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

# =======================
# LOSSES
# =======================
def bce_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7), tf.float32)
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

def focal_bce(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7), tf.float32)
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
    focal_weight = alpha_t * tf.pow(1.0 - p_t, gamma)
    return tf.reduce_mean(focal_weight * bce)

def tversky_loss(y_true, y_pred, alpha=0.70, beta=0.30, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7), tf.float32)
    tp = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=[1,2,3])
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=[1,2,3])
    t = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tf.reduce_mean(t)

def total_loss(y_true, y_pred, cfg):
    lt = tversky_loss(y_true, y_pred, alpha=cfg["tv_alpha"], beta=cfg["tv_beta"])
    lb = focal_bce(y_true, y_pred) if cfg["use_focal"] else bce_loss(y_true, y_pred)
    return float(cfg["tv_w"]) * lt + float(cfg["bce_w"]) * lb

# =======================
# MODEL
# =======================
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

def _dec(x, skip, f, drop, idx=0):
    x = Conv2DTranspose(f, 2, strides=2, padding="same", name=f"up_{idx}")(x)
    x = _safe_concat(x, skip, idx=idx)
    x = _conv(x, f, 3, drop=drop)
    x = _conv(x, f, 3, drop=0.0)
    return x

def build_unet_effb0(net_h, net_w, out_channels=1, train_encoder=False):
    inp = Input((net_h, net_w, 3))
    x0 = Lambda(lambda t: t * 255.0, name="scale_255")(inp)
    x0 = efficientnet.preprocess_input(x0)

    base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x0)
    base.trainable = bool(train_encoder)

    s1 = base.get_layer("stem_activation").output
    s2 = base.get_layer("block2a_activation").output
    s3 = base.get_layer("block3a_activation").output
    s4 = base.get_layer("block4a_activation").output
    b  = base.get_layer("top_activation").output

    b = _conv(b, 256, 3, drop=BOT_DROPOUT)
    b = _conv(b, 256, 3, drop=0.0)

    d4 = _dec(b,  s4, 256, drop=DEC_DROPOUT, idx=4)
    d3 = _dec(d4, s3, 128, drop=DEC_DROPOUT, idx=3)
    d2 = _dec(d3, s2, 64,  drop=DEC_DROPOUT, idx=2)
    d1 = _dec(d2, s1, 32,  drop=DEC_DROPOUT, idx=1)

    x = Conv2DTranspose(16, 2, strides=2, padding="same", name="up_final")(d1)
    x = _conv(x, 16, 3, drop=0.0)

    out = Conv2D(out_channels, 1, activation="sigmoid", dtype="float32", name="mask")(x)
    return Model(inp, out, name="UNet_EfficientNetB0"), base

def freeze_all_bn(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

# =======================
# OVERLAY GRID
# =======================
def _to_uint8_rgb(x01):
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0 + 0.5).astype(np.uint8)

def _mask_to_rgba(mask01, color=(255, 0, 0), alpha=120):
    m = (mask01 > 0.5).astype(np.uint8) * alpha
    H, W = m.shape[:2]
    r = np.full((H, W), color[0], np.uint8)
    g = np.full((H, W), color[1], np.uint8)
    b = np.full((H, W), color[2], np.uint8)
    a = m.astype(np.uint8)
    return np.stack([r, g, b, a], axis=-1)

def _render_overlay_tile(base_rgb01: np.ndarray, mask01: np.ndarray, color, alpha, text: str):
    base = Image.fromarray(_to_uint8_rgb(base_rgb01), mode="RGB").convert("RGBA")
    ov = Image.fromarray(_mask_to_rgba(mask01, color=color, alpha=alpha), mode="RGBA")
    out = Image.alpha_composite(base, ov).convert("RGB")
    ImageDraw.Draw(out).text((6, 6), text, fill=(255, 255, 255))
    return out

def _select_indices_grouped_global4(pairs, n_pairs, seed):
    rng = np.random.RandomState(seed)
    n_pairs = int(n_pairs)

    buckets = {c: [] for c in CLASS_NAMES}
    other = []
    for i, (cls, _, _) in enumerate(pairs):
        if cls in buckets:
            buckets[cls].append(i)
        else:
            other.append(i)

    for c in CLASS_NAMES:
        rng.shuffle(buckets[c])
    rng.shuffle(other)

    have = [c for c in CLASS_NAMES if len(buckets[c]) > 0]
    if not have:
        all_idx = list(range(len(pairs)))
        rng.shuffle(all_idx)
        groups = {"UNKNOWN": all_idx[:min(n_pairs, len(all_idx))]}
        return groups, ["UNKNOWN"]

    base_quota = 1
    remaining_budget = max(0, n_pairs - min(n_pairs, len(have)) )
    extra_each = remaining_budget // max(1, len(have))
    extra_rem  = remaining_budget %  max(1, len(have))

    quotas = {c: 0 for c in CLASS_NAMES}

    for c in have:
        if n_pairs > 0:
            quotas[c] = base_quota

    for c in have:
        quotas[c] += extra_each

    for c in [c for c in CLASS_NAMES if c in have][:extra_rem]:
        quotas[c] += 1

    groups = {c: [] for c in CLASS_NAMES}
    used = set()
    for c in CLASS_NAMES:
        q = int(quotas.get(c, 0))
        take = buckets[c][:q]
        for idx in take:
            groups[c].append(idx)
            used.add(idx)

    total = sum(len(v) for v in groups.values())
    if total < n_pairs:
        leftovers = []
        for c in CLASS_NAMES:
            for idx in buckets[c]:
                if idx not in used:
                    leftovers.append((c, idx))
        rng.shuffle(leftovers)

        for c, idx in leftovers:
            if total >= n_pairs:
                break
            groups[c].append(idx)
            used.add(idx)
            total += 1

    total = sum(len(v) for v in groups.values())
    if total > n_pairs:
        over = total - n_pairs
        for c in reversed(CLASS_NAMES):
            while over > 0 and groups[c]:
                groups[c].pop()
                over -= 1
            if over <= 0:
                break

    groups = {c: groups[c] for c in CLASS_NAMES if groups[c]}
    return groups, [c for c in CLASS_NAMES if c in groups]

def save_val_overlay_grid(model, pairs, args, out_png: str, n_pairs: int = 12, 
    thr: float = 0.5, seed: int = GLOBAL_SEED, cols: int = 4,):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    if len(pairs) == 0:
        print("[WARN] no pairs for overlay grid")
        return

    rng = np.random.RandomState(seed)
    n_pairs = int(n_pairs)
    tiles = []
    class_headers = []

    groups, order = _select_indices_grouped_global4(pairs, n_pairs=n_pairs, seed=seed)

    cov = {c: len(groups.get(c, [])) for c in CLASS_NAMES}
    print("[OVERLAY_GRID] class coverage:", cov)

    for cname in order:
        idx_list = groups[cname]
        if not idx_list:
            continue

        class_headers.append((cname, len(tiles)))

        for i in idx_list:
            cls, ip, mp = pairs[i]
            x = read_image(ip, args.img_h, args.img_w)
            y_bin = read_mask_binary(mp, args.img_h, args.img_w)[..., 0]
            pred = model.predict(np.expand_dims(x, 0), verbose=0)[0]

            ci = CLASS_TO_INDEX.get(cls, None)
            if ci is None:
                continue
            pred_bin = (pred[..., ci] >= float(thr)).astype(np.float32)

            stem = Path(ip).stem
            title = f"{cls} | ch={ci}"
            common = f"{title} | {stem} | thr={thr:.2f}"
            tiles.append(_render_overlay_tile(x, y_bin,  color=(0, 255, 0), alpha=130, text=f"GT  | {common}"))
            tiles.append(_render_overlay_tile(x, pred_bin, color=(255, 0, 0), alpha=130, text=f"PRED| {common}"))

    if not tiles:
        print("[WARN] no tiles produced.")
        return []

    # Slide-friendly output: 2x2 grid (4 tiles) per page.
    items_per_page = 4
    page_cols = 2
    page_rows = 2
    tile_w, tile_h = tiles[0].size
    pad = 6
    saved_pages = []
    base_no_ext, ext = os.path.splitext(out_png)
    if not ext:
        ext = ".png"

    total_pages = int(np.ceil(len(tiles) / items_per_page))
    for page_idx in range(total_pages):
        start = page_idx * items_per_page
        end = min(start + items_per_page, len(tiles))
        page_tiles = tiles[start:end]

        grid_w = page_cols * tile_w + (page_cols + 1) * pad
        grid_h = page_rows * tile_h + (page_rows + 1) * pad
        grid = Image.new("RGB", (grid_w, grid_h), color=(20, 20, 20))

        for k, tile in enumerate(page_tiles):
            r = k // page_cols
            c = k % page_cols
            x0 = pad + c * (tile_w + pad)
            y0 = pad + r * (tile_h + pad)
            grid.paste(tile, (x0, y0))

        page_path = f"{base_no_ext}_page_{page_idx + 1:02d}{ext}"
        grid.save(page_path)
        saved_pages.append(page_path)

    print(f"[OK] saved overlay grid pages -> {base_no_ext}_page_XX{ext} | pages={len(saved_pages)} | tiles={len(tiles)} (pairs~={len(tiles)//2})")
    return saved_pages

# =======================
# EVAL DETAIL
# =======================
def pixel_confusion(y_true01: np.ndarray, y_prob01: np.ndarray, thr: float = 0.5):
    yt = (y_true01 >= 0.5).astype(np.uint8).reshape(-1)
    yp = (y_prob01 >= float(thr)).astype(np.uint8).reshape(-1)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return tn, fp, fn, tp

def cm_report(tn: int, fp: int, fn: int, tp: int):
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    specificity = tn / (tn + fp + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "specificity": float(specificity),
        "iou": float(iou),
        "dice": float(dice),
        "support_pos": int(tp + fn),
        "support_neg": int(tn + fp),
    }

def plot_multiclass_confusion_matrix(cm: np.ndarray, class_names: list, out_path: str, 
                                     title="Multi-Class Confusion Matrix", normalize=False):
    mat = cm.astype(np.float64)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        mat = mat / np.maximum(row_sums, 1e-12)
    
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(mat, interpolation='nearest', cmap='Blues', vmin=0)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Count' if normalize else 'Count', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    
    # Add text annotations
    thresh = mat.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = mat[i, j]
            if normalize:
                text = f"{val:.2f}"
            else:
                # Format with K/M for large numbers
                if val >= 1e6:
                    text = f"{val/1e6:.1f}M"
                elif val >= 1e3:
                    text = f"{val/1e3:.1f}K"
                else:
                    text = f"{int(val)}"
            
            text_color = "white" if val > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", 
                   color=text_color, fontsize=9, weight='bold')
    
    # Labels and title
    ax.set_ylabel('True Label', fontsize=12, weight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    
    # Calculate and display overall accuracy
    total = cm.sum()
    correct = np.diag(cm).sum()
    accuracy = correct / max(total, 1)
    
    metrics_text = f"Overall Pixel Accuracy: {accuracy:.4f} ({correct:,} / {int(total):,})"
    plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] saved multi-class confusion matrix -> {out_path}")

# =======================
# TUNE / TRAIN / EVAL
# =======================
def make_candidates(args, space: dict, ab_tol=1e-6, max_random_tries=200000):
    keys = list(space.keys())
    uniq = {}

    def cast(raw):
        c2 = dict(raw)
        c2["train_encoder"] = bool(int(c2["train_encoder"])) if isinstance(c2["train_encoder"], (int, np.integer)) else bool(c2["train_encoder"])
        c2["use_focal"] = bool(int(c2["use_focal"])) if isinstance(c2["use_focal"], (int, np.integer)) else bool(c2["use_focal"])
        c2 = normalize_cfg(c2)
        c2 = apply_rosette_policy(args, c2)
        return c2

    method = args.search_method
    if method == "grid":
        raw_count = 1
        for k in keys:
            raw_count *= max(1, len(space[k]))
        if raw_count > 5000:
            print(f"[WARN] Grid cartesian = {raw_count} kombinasi (sebelum filter alpha+beta==1).")

        for combo in itertools.product(*[space[k] for k in keys]):
            raw = dict(zip(keys, combo))
            if abs((float(raw["tv_alpha"]) + float(raw["tv_beta"])) - 1.0) > ab_tol:
                continue
            c = cast(raw)
            uniq[cfg_key(c)] = c
    elif method == "random":
        domain = {k: list(space[k]) for k in keys}
        target = int(args.tune_trials)
        tries = 0
        while len(uniq) < target and tries < max_random_tries:
            tries += 1
            raw = {k: random.choice(domain[k]) for k in keys}
            if abs((float(raw["tv_alpha"]) + float(raw["tv_beta"])) - 1.0) > ab_tol:
                continue
            c = cast(raw)
            uniq[cfg_key(c)] = c
        if len(uniq) < target:
            print(f"[WARN] valid random candidates hanya {len(uniq)}/{target}. Pastikan alpha/beta komplementer.")
    else:
        raise ValueError("search_method must be grid or random")

    return list(uniq.values())

def run_proxy_trial(val_pairs, args, cfg):
    tf.keras.backend.clear_session()

    cfg = normalize_cfg(cfg)
    cfg = apply_rosette_policy(args, cfg)
    set_dropout_globals(cfg)

    out_ch = len(CLASS_NAMES)
    tune_bs = int(args.tune_batch_size)

    ds_tr, _ = build_train_dataset_mix(args, cfg, batch=tune_bs)
    ds_va = make_dataset(val_pairs, args, batch=tune_bs, shuffle=False)

    model, _ = build_unet_effb0(args.img_h, args.img_w, out_channels=out_ch, train_encoder=bool(cfg["train_encoder"]))
    model.build((None, args.img_h, args.img_w, 3))

    if args.freeze_bn:
        freeze_all_bn(model)

    loss_fn = lambda yt, yp: total_loss(yt, yp, cfg)
    metrics = [DiceBin(0.5, "dice50"), IoUBin(0.5, "iou50")]
    opt = tf.keras.optimizers.Adam(learning_rate=float(cfg["lr"]), clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

    es = EarlyStopping(
        monitor="val_loss", mode="min",
        patience=int(args.tune_patience),
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=0
    )

    fit_kwargs = dict(epochs=int(args.tune_epochs), callbacks=[es], verbose=0)
    if int(args.tune_val_steps) > 0:
        fit_kwargs["validation_steps"] = int(args.tune_val_steps)

    model.fit(ds_tr, validation_data=ds_va, **fit_kwargs)

    hist = model.history.history
    best_val_loss = float(np.min(hist.get("val_loss", [np.inf])))
    best_val_dice = float(np.max(hist.get("val_dice50", [-1.0])))

    cleanup(model, opt, ds_tr, ds_va)
    return best_val_loss, best_val_dice

def tune_mode(args):
    val_root = os.path.join(BASE_DIR, args.val_split)
    val_pairs = collect_pairs(val_root, args)
    if len(val_pairs) == 0:
        raise RuntimeError("Val pairs kosong. Cek val_split.")

    grid_mix = args.grid_mix

    space = {
        "lr": args.grid_lr,
        "mix_full_ratio": grid_mix,
        "train_encoder": args.grid_train_encoder,
        "tv_w": args.grid_tv_w,
        "tv_alpha": args.grid_tv_alpha,
        "tv_beta": args.grid_tv_beta,
        "use_focal": args.grid_use_focal,
        "bot_dropout": args.grid_bot_dropout,
        "dec_dropout": args.grid_dec_dropout,
    }

    cands = make_candidates(args, space)
    if len(cands) == 0:
        raise RuntimeError("Tidak ada candidate valid setelah filter alpha+beta==1.")

    out_cfg = cfg_path(args)
    out_csv = trials_csv_path(args)

    print(f"\nTuning {args.search_method}: {len(cands)} candidate(s) | objective=MIN val_loss")
    print(f"[TUNE] tune_batch_size={args.tune_batch_size} | tune_val_steps={args.tune_val_steps}")
    print(f"[TUNE] output cfg -> {out_cfg}")

    rows = []
    for i, cfg in enumerate(cands, start=1):
        print(f"  Trial {i}/{len(cands)} -> {cfg}")
        vloss, vdice = run_proxy_trial(val_pairs, args, cfg)
        rows.append({**cfg, "best_val_loss": vloss, "best_val_dice": vdice})

    rows_sorted = sorted(rows, key=lambda r: float(r["best_val_loss"]))

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)
    print(f"Saved tuning trials -> {out_csv}")

    best = rows_sorted[0]
    best_cfg = {k: best[k] for k in [
        "mix_full_ratio","lr","train_encoder",
        "tv_w","bce_w","tv_alpha","tv_beta","use_focal",
        "bot_dropout","dec_dropout"
    ]}
    best_cfg = normalize_cfg(best_cfg)
    best_cfg = apply_rosette_policy(args, best_cfg)

    jsave(out_cfg, dict(best_cfg))
    print(f"Saved best tuned cfg -> {out_cfg}")

def train_mode(args):
    cfg = load_cfg(args)
    cfg = apply_rosette_policy(args, cfg)
    set_dropout_globals(cfg)

    ds_tr, info = build_train_dataset_mix(args, cfg, batch=int(args.batch_size))
    val_root = os.path.join(BASE_DIR, args.val_split)
    val_pairs = collect_pairs(val_root, args)
    if len(val_pairs) == 0:
        raise RuntimeError("Val pairs kosong. Cek val_split.")
    ds_va = make_dataset(val_pairs, args, batch=int(args.batch_size), shuffle=False)

    out_ch = len(CLASS_NAMES)
    model, _ = build_unet_effb0(args.img_h, args.img_w, out_channels=out_ch, train_encoder=bool(cfg["train_encoder"]))
    model.build((None, args.img_h, args.img_w, 3))

    if bool(cfg.get("freeze_bn", True)) or args.freeze_bn:
        freeze_all_bn(model)

    loss_fn = lambda yt, yp: total_loss(yt, yp, cfg)
    metrics = [DiceBin(0.5, "dice50"), IoUBin(0.5, "iou50")]

    weights_path, hist_path, spec_path, tag = artifact_paths(args)

    monitor = args.monitor_metric
    mode = "min" if monitor == "val_loss" else "max"

    ckpt = ModelCheckpoint(weights_path, monitor=monitor, mode=mode, save_best_only=True, save_weights_only=True, verbose=1)
    es   = EarlyStopping(monitor=monitor, mode=mode, patience=args.es_patience, restore_best_weights=True, verbose=1)
    rlr  = ReduceLROnPlateau(monitor=monitor, mode=mode, factor=0.5, patience=args.lr_patience, min_lr=args.min_lr, verbose=1)

    print("\n[TRAIN_SETUP]")
    print(f"  run            : GLOBAL4")
    print(f"  data_mode      : {info.get('mode')}")
    print(f"  input          : {args.img_h}x{args.img_w} | bs={args.batch_size}")
    print(f"  cfg            : {cfg}")
    print(f"  monitor        : {monitor} ({mode})")
    print(f"  ckpt(best)     : {weights_path}")

    opt = tf.keras.optimizers.Adam(learning_rate=float(cfg["lr"]), clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

    hist = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, callbacks=[ckpt, es, rlr], verbose=1)
    hist_clean = {k: [float(x) for x in v] for k, v in hist.history.items()}
    jsave(hist_path, hist_clean)
    print(f"[OK] saved history -> {hist_path}")

    prefix = f"best_{tag}_segmentation_model"
    plot_training_curves(hist_clean, out_dir=MODEL_DIR, prefix=prefix)

    spec = {
        "model": "UNet_EfficientNetB0",
        "img_size": [int(args.img_h), int(args.img_w)],
        "run_type": "global4",
        "only_class": None,
        "class_names": CLASS_NAMES,
        "selected_cfg": cfg,
        "cfg_source": args.cfg_json if args.cfg_json else cfg_path(args),
        "monitor": {"metric": monitor, "mode": mode},
        "constraints": {"tv_w+bce_w": 1.0, "tv_alpha+tv_beta": 1.0},
        "paths": {
            "best_weights": weights_path,
            "history_json": hist_path,
            "spec_json": spec_path,
            "curve_loss_png": os.path.join(MODEL_DIR, f"{prefix}_curve_loss.png"),
            "curve_dice_png": os.path.join(MODEL_DIR, f"{prefix}_curve_dice.png"),
            "curve_iou_png":  os.path.join(MODEL_DIR, f"{prefix}_curve_iou.png"),
        }
    }
    jsave(spec_path, spec)
    print(f"[OK] saved spec -> {spec_path}")

    if args.save_overlays:
        rr = res_root(args)
        os.makedirs(rr, exist_ok=True)
        folder_tag = safe_name(args.tag.strip() if args.tag.strip() else "GLOBAL4")
        out_png = os.path.join(
            rr,
            f"{slug(folder_tag)}_train_overlay_grid_thr{args.overlay_thr:.2f}".replace(".", "p") + ".png"
        )
        save_val_overlay_grid(
            model=model,
            pairs=val_pairs,
            args=args,
            out_png=out_png,
            n_pairs=args.overlay_n,
            thr=args.overlay_thr,
            seed=GLOBAL_SEED,
            cols=args.overlay_cols
        )

    cleanup(model, opt, ds_tr, ds_va)

def eval_mode(args):
    tf.keras.backend.clear_session()

    cfg = load_cfg(args)
    cfg = apply_rosette_policy(args, cfg)
    set_dropout_globals(cfg)

    val_root = os.path.join(BASE_DIR, args.val_split)
    val_pairs = collect_pairs(val_root, args)
    if len(val_pairs) == 0:
        raise RuntimeError("Val pairs kosong. Cek val_split.")

    ds = make_dataset(val_pairs, args, batch=int(args.batch_size), shuffle=False)

    out_ch = len(CLASS_NAMES)
    model, _ = build_unet_effb0(args.img_h, args.img_w, out_channels=out_ch, train_encoder=False)
    model.build((None, args.img_h, args.img_w, 3))
    model.load_weights(args.weights)

    if args.freeze_bn:
        freeze_all_bn(model)

    loss_fn = lambda yt, yp: total_loss(yt, yp, cfg)
    metrics = [DiceBin(0.5, "dice50"), IoUBin(0.5, "iou50")]
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_fn, metrics=metrics)

    out = model.evaluate(ds, verbose=1, return_dict=True)
    out_clean = {k: float(v) for k, v in out.items()}
    print("[EVAL]", json.dumps(out_clean, indent=2))

    rr = res_root(args)
    os.makedirs(rr, exist_ok=True)
    
    if args.cm_enable:
        thr_cm = float(args.cm_thr)
        print(f"\n[CONFUSION_MATRIX] pixel-level @ thr={thr_cm:.3f}")

        ys, ps = [], []
        for xb, yb in ds:
            pb = model.predict(xb, verbose=0)
            ys.append(yb.numpy())
            ps.append(pb)
        y_true = np.concatenate(ys, axis=0)
        y_prob = np.concatenate(ps, axis=0)

        reports = []
        # Build multi-class confusion matrix (5x5: Background + 4 disease classes)
        # Convert predictions to class indices
        y_true_flat = y_true.reshape(-1, len(CLASS_NAMES))  # (N_pixels, 4)
        y_prob_flat = y_prob.reshape(-1, len(CLASS_NAMES))  # (N_pixels, 4)
        
        # Determine true and predicted classes
        # Class 0 = Background (all channels < 0.5)
        # Class 1-4 = Disease classes (argmax of channels that are >= threshold)
        y_true_class = np.zeros(y_true_flat.shape[0], dtype=np.int32)
        y_pred_class = np.zeros(y_prob_flat.shape[0], dtype=np.int32)
        
        # For true labels: if any channel is > 0.5, assign to that class (1-indexed), else background (0)
        true_max = np.max(y_true_flat, axis=1)
        true_has_lesion = true_max > 0.5
        y_true_class[true_has_lesion] = np.argmax(y_true_flat[true_has_lesion], axis=1) + 1
        
        # For predictions: if any channel is >= threshold, assign to that class, else background
        pred_max = np.max(y_prob_flat, axis=1)
        pred_has_lesion = pred_max >= thr_cm
        y_pred_class[pred_has_lesion] = np.argmax(y_prob_flat[pred_has_lesion], axis=1) + 1
        
        # Build 5x5 confusion matrix
        n_classes = len(CLASS_NAMES) + 1  # +1 for background
        cm_multi = np.zeros((n_classes, n_classes), dtype=np.int64)
        for true_c, pred_c in zip(y_true_class, y_pred_class):
            cm_multi[true_c, pred_c] += 1
        
        # Save normalized multi-class confusion matrix visualization
        all_class_names = ["Background"] + CLASS_NAMES
        cm_multi_norm_png = os.path.join(rr, f"cm_multiclass_normalized_{Path(args.weights).stem}_thr{thr_cm:.2f}.png".replace(".", "p"))
        plot_multiclass_confusion_matrix(
            cm=cm_multi,
            class_names=all_class_names,
            out_path=cm_multi_norm_png,
            title=f"Multi-Class Confusion Matrix - Normalized (thr={thr_cm:.2f})",
            normalize=True
        )
        
        # Calculate per-class metrics for reports (display only)
        for ci, cname in enumerate(CLASS_NAMES):
            tn, fp, fn, tp = pixel_confusion(y_true[..., ci], y_prob[..., ci], thr=thr_cm)
            rep = cm_report(tn, fp, fn, tp)
            print(f"  {cname}: TP={tp} FP={fp} FN={fn} TN={tn} | P={rep['precision']:.4f} R={rep['recall']:.4f} F1={rep['f1']:.4f}")

    if args.save_overlays:
        out_png = os.path.join(
            rr,
            f"eval_{Path(args.weights).stem}_overlay_grid_thr{args.overlay_thr:.2f}".replace(".", "p") + ".png"
        )
        save_val_overlay_grid(
            model=model,
            pairs=val_pairs,
            args=args,
            out_png=out_png,
            n_pairs=args.overlay_n,
            thr=args.overlay_thr,
            seed=GLOBAL_SEED,
            cols=args.overlay_cols
        )

    cleanup(model, ds)

# =======================
# CLI
# =======================
def parse_args():
    p = argparse.ArgumentParser(description="Tune/Train/Eval U-Net EffNetB0 segmentation (simplified; outputs preserved)")
    p.add_argument("--mode", choices=["tune", "train", "eval"], default="train")

    p.add_argument("--cfg_json", default="models/segmentation/best_tuned_cfg.json", help="Optional cfg json; if empty uses best_tuned_cfg.json")

    p.add_argument("--img_h", type=int, default=480)
    p.add_argument("--img_w", type=int, default=640)

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--lr_patience", type=int, default=6)
    p.add_argument("--es_patience", type=int, default=16)
    p.add_argument("--monitor_metric", default="val_dice50", choices=["val_loss", "val_dice50", "val_iou50"])

    p.add_argument("--train_roi_split",  default="train_roi")
    p.add_argument("--train_full_split", default="train_balanced_perclass")
    p.add_argument("--val_split", default="val")

    p.add_argument("--weights", default="")
    p.add_argument("--save_overlays", action="store_true")
    p.add_argument("--overlay_n", type=int, default=12, help="pairs sampled; tiles = 2*overlay_n")
    p.add_argument("--overlay_thr", type=float, default=0.50)
    p.add_argument("--overlay_cols", type=int, default=4)
    p.add_argument("--tag", default="")
    p.add_argument("--cm_enable", action="store_true")
    p.add_argument("--cm_thr", type=float, default=0.50)
    p.add_argument("--freeze_bn", action="store_true")

    p.add_argument("--search_method", choices=["grid", "random"], default="random")
    p.add_argument("--tune_trials", type=int, default=30)
    p.add_argument("--tune_epochs", type=int, default=15)
    p.add_argument("--tune_patience", type=int, default=6)
    p.add_argument("--tune_batch_size", type=int, default=1)
    p.add_argument("--tune_val_steps", type=int, default=0)

    p.add_argument("--grid_lr", type=float, nargs="+", default=[5e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4])
    p.add_argument("--grid_mix", type=float, nargs="+", default=[0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    p.add_argument("--grid_train_encoder", type=int, nargs="+", default=[0, 1])
    p.add_argument("--grid_tv_w", type=float, nargs="+", default=[0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90])
    p.add_argument("--grid_tv_alpha", type=float, nargs="+", default=[0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90])
    p.add_argument("--grid_tv_beta", type=float, nargs="+", default=[0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90])
    p.add_argument("--grid_use_focal", type=int, nargs="+", default=[0, 1])
    p.add_argument("--grid_bot_dropout", type=float, nargs="+", default=[0.00, 0.03, 0.06, 0.10, 0.14])
    p.add_argument("--grid_dec_dropout", type=float, nargs="+", default=[0.00, 0.02, 0.04, 0.06, 0.08, 0.10])

    args = p.parse_args()
    args.cfg_json = (args.cfg_json or "").strip()
    args.tag = args.tag or ""

    if args.mode == "eval" and not args.weights:
        raise RuntimeError("--weights wajib untuk mode eval")

    return args

if __name__ == "__main__":
    setup()
    args = parse_args()
    if args.mode == "tune":
        tune_mode(args)
    elif args.mode == "train":
        train_mode(args)
    else:
        eval_mode(args)