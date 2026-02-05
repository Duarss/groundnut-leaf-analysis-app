# ==========================================
# exe_severity_model.py
# ==========================================
# Binary leaf-object segmentation (severity leaf mask)
# TF/Keras 2.10.0 (Windows GPU)
#
# Pipeline:
#   - tune   : manual random search (objective: val_loss)
#   - train  : train best config, save weights only (+ checkpoint best-per-epoch)
#   - eval   : evaluate on val set, save predicted mask + masked image (img * predmask)
#             + (SIDANG) save eval summary JSON/CSV + per-image metrics CSV
#             + (SIDANG) optional pixel-level confusion-matrix report (--cm_enable)
#   - predict: single image inference (supports GUI file picker)
#
# Dataset structure:
#   severity_dataset/
#     train/
#       images/
#       masks/   (binary png 0/255)
#     val/
#       images/
#       masks/
#
# Artifacts:
#   - best_severity_config.json
#   - severity_tuning_log.csv
#   - best_severity_model.weights.h5       (BEST weights)  <-- renamed
#   - final_severity_model.weights.h5            (LAST epoch)
#   - severity_train_history.csv
#   - severity_train_history.json
#   - severity_curve_loss.png
#   - severity_curve_dice.png
#   - severity_curve_iou.png
#   - eval_outputs/ (inside out_dir by default)
#       eval_summary_*.json
#       eval_summary_*.csv
#       eval_per_image_*.csv
#       eval_metrics_*.png
#
# Notes:
# - Default input size is 480x640 (landscape) to align downstream severity calc.
# - GPU 4GB: default batch_size=1. Trials that OOM will be skipped.
#
# Example:
#   python exe_severity_model.py tune    --data_root utils/severity_dataset --out_dir models/severity/leaf_mask --trials 30 --epochs 8
#   python exe_severity_model.py train   --data_root utils/severity_dataset --out_dir models/severity/leaf_mask --epochs 40
#   python exe_severity_model.py eval    --data_root utils/severity_dataset --out_dir results/severity/leaf_mask --threshold 0.5 --cm_enable
#   python exe_severity_model.py predict --weights models/severity/leaf_mask/final_severity_model.weights.h5 --image path/to/img.jpg --threshold 0.5
#   python exe_severity_model.py predict --weights models/severity/leaf_mask/final_severity_model.weights.h5 --gui
# ==========================================

import argparse
import csv
import gc
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ========= Paths (mengikuti pola exe_segmentation_model.py) =========
BASE_DIR  = "utils/severity_dataset"
MODEL_DIR = "models/severity/leaf_mask"
RES_DIR   = "results/severity/leaf_mask"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# Output default (bisa override via CLI)
DEFAULT_OUT_DIR = MODEL_DIR

# ----------------------------
# Repro & GPU util
# ----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def set_memory_growth():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

def enable_mixed_precision(enable: bool):
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16" if enable else "float32")
    except Exception:
        pass


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Plotting helpers (paper-ready)
# ----------------------------
def plot_training_curves(history_dict, out_dir, prefix="severity"):
    """
    Buat 3 grafik:
      1) loss: train loss vs val loss
      2) dice50: train dice50 vs val_dice50 (jika ada)
      3) iou50 : train iou50  vs val_iou50  (jika ada)
    """
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    if not history_dict or not isinstance(history_dict, dict):
        print("[WARN] history kosong. Skip plotting.")
        return

    n_epochs = None
    for k, v in history_dict.items():
        if isinstance(v, (list, tuple)) and len(v) > 0:
            n_epochs = len(v)
            break
    if not n_epochs:
        print("[WARN] history tidak punya epoch data. Skip plotting.")
        return

    epochs = np.arange(1, n_epochs + 1)

    def _plot_pair(train_key, val_key, ylabel, title, filename, lt, lv):
        if train_key not in history_dict:
            return
        plt.figure()
        plt.plot(epochs, history_dict.get(train_key, []), label=lt)
        if val_key in history_dict:
            plt.plot(epochs, history_dict.get(val_key, []), label=lv)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        p = out_dir / filename
        plt.savefig(str(p), dpi=200)
        plt.close()
        print("[OK] saved plot:", p)

    _plot_pair("loss", "val_loss", "Loss", "Training vs Validation Loss",
               f"{prefix}_curve_loss.png", "train_loss", "val_loss")

    _plot_pair("dice50", "val_dice50", "Dice@0.5", "Training vs Validation Dice@0.5",
               f"{prefix}_curve_dice.png", "train_dice50", "val_dice50")

    _plot_pair("iou50", "val_iou50", "IoU@0.5", "Training vs Validation IoU@0.5",
               f"{prefix}_curve_iou.png", "train_iou50", "val_iou50")


def plot_eval_metrics_bar(metrics_dict, out_dir, prefix="eval_metrics"):
    """Bar plot sederhana untuk (loss, dice50, iou50) agar siap masuk paper."""
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    keys_pref = ["loss", "dice50", "iou50"]
    keys = [k for k in keys_pref if k in metrics_dict]
    if not keys:
        return

    vals = [float(metrics_dict[k]) for k in keys]
    plt.figure(figsize=(7, 3.5))
    plt.bar(range(len(keys)), vals)
    plt.xticks(range(len(keys)), keys, rotation=20, ha="right")
    plt.ylabel("Value")
    plt.title("Evaluation Metrics")
    plt.tight_layout()
    p = out_dir / f"{prefix}.png"
    plt.savefig(str(p), dpi=200)
    plt.close()
    print("[OK] saved eval plot:", p)


# ----------------------------
# IO helpers
# ----------------------------
def list_images(img_dir: Path):
    out = []
    for p in sorted(img_dir.glob("*")):
        if p.suffix.lower() in VALID_EXT:
            out.append(str(p))
    return out

def find_image_by_stem(paths, stem):
    for p in paths:
        if Path(p).stem == stem:
            return p
    return None

def build_pairs(img_dir: Path, mask_dir: Path):
    if not img_dir.exists() or not mask_dir.exists():
        return []
    imgs = list_images(img_dir)
    pairs = []
    for ip in imgs:
        stem = Path(ip).stem
        mp = None
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
            cand = mask_dir / f"{stem}{ext}"
            if cand.exists():
                mp = str(cand)
                break
        if mp is not None:
            pairs.append((ip, mp))
    return pairs

def _to_builtin(obj):
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def _save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_to_builtin)

def pick_image_file_gui(title="Select image"):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(title=title)
        root.update()
        root.destroy()
        return path
    except Exception:
        return None


# ----------------------------
# Dataset pipeline
# ----------------------------
def decode_image(path, out_h, out_w):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    x = bgr.astype(np.float32) / 255.0
    return x

def preprocess_pair(img_path, mask_path, out_h, out_w):
    x = decode_image(img_path, out_h, out_w)
    if x is None:
        return None, None

    m0 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m0 is None:
        return None, None
    m = cv2.resize(m0, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    y = (m > 0).astype(np.float32)[..., None]
    return x, y

def make_dataset(pairs, out_h, out_w, batch_size, shuffle=False, seed=42):
    def gen():
        idx = np.arange(len(pairs))
        if shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(idx)
        for i in idx:
            ip, mp = pairs[i]
            x, y = preprocess_pair(ip, mp, out_h, out_w)
            if x is None:
                continue
            yield x, y

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(out_h, out_w, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(out_h, out_w, 1), dtype=tf.float32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(128, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ----------------------------
# Losses
# ----------------------------
def soft_dice_for_loss(y_true, y_pred, eps=1e-7):
    """
    Dice versi 'soft' (tanpa threshold) untuk LOSS agar differentiable.
    Metric untuk laporan tetap pakai DiceBin/IoUBin (thresholded).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    den = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    dice = (2.0 * inter + 1.0) / (den + 1.0 + eps)
    return dice

def dice_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(soft_dice_for_loss(y_true, y_pred))

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7), tf.float32)
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
    weight = alpha_t * tf.pow(1.0 - p_t, gamma)
    return tf.reduce_mean(weight * bce)

def composite_loss(y_true, y_pred, w_dice=0.7, w_bce=0.3, use_focal=False):
    ld = dice_loss(y_true, y_pred)
    if use_focal:
        lb = focal_loss(y_true, y_pred)
    else:
        lb = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return float(w_dice) * ld + float(w_bce) * lb


# ----------------------------
# Model: simple U-Net (configurable)
# ----------------------------
def conv_block(x, f, use_bn=True, dropout=0.0, sep_conv=False):
    Conv = tf.keras.layers.SeparableConv2D if sep_conv else tf.keras.layers.Conv2D
    x = Conv(f, 3, padding="same")(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    if dropout and dropout > 0:
        x = tf.keras.layers.SpatialDropout2D(dropout)(x)
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

    out = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", dtype="float32")(x)
    return tf.keras.Model(inp, out)


def compile_model(model, cfg):
    lr = float(cfg.get("lr", 1e-3))
    optimizer_name = str(cfg.get("optimizer", "adam")).lower()

    if optimizer_name == "adam":
        opt = tf.keras.optimizers.Adam(lr)
    elif optimizer_name == "adamw":
        opt = tf.keras.optimizers.AdamW(lr)
    else:
        opt = tf.keras.optimizers.Adam(lr)

    model.compile(
        optimizer=opt,
        loss=lambda yt, yp: composite_loss(
            yt,
            yp,
            w_dice=float(cfg.get("w_dice", 0.7)),
            w_bce=float(cfg.get("w_bce", 0.3)),
            use_focal=bool(cfg.get("use_focal", False)),
        ),
        metrics=[DiceBin(threshold=0.5, name="dice50"), IoUBin(threshold=0.5, name="iou50")],
    )
    return model


# ----------------------------
# Metrics helpers (for eval & train)
# ----------------------------
class DiceBin(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name="dice50", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.sum = self.add_weight(name="sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true > 0.5, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        den = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
        dice = (2.0 * inter + 1.0) / (den + 1.0 + 1e-9)
        self.sum.assign_add(tf.reduce_sum(dice))
        self.count.assign_add(tf.cast(tf.size(dice), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.sum, self.count)

    def reset_state(self):
        self.sum.assign(0.0)
        self.count.assign(0.0)


class IoUBin(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name="iou50", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.sum = self.add_weight(name="sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true > 0.5, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        union = tf.reduce_sum(tf.cast((y_true + y_pred) > 0, tf.float32), axis=[1, 2, 3])
        iou = (inter + 1.0) / (union + 1.0 + 1e-9)
        self.sum.assign_add(tf.reduce_sum(iou))
        self.count.assign_add(tf.cast(tf.size(iou), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.sum, self.count)

    def reset_state(self):
        self.sum.assign(0.0)
        self.count.assign(0.0)


# ----------------------------
# Confusion-matrix helpers
# ----------------------------
def cm_from_binary(gt01: np.ndarray, pr01: np.ndarray):
    gt01 = gt01.astype(np.uint8).reshape(-1)
    pr01 = pr01.astype(np.uint8).reshape(-1)
    tp = int(np.sum((gt01 == 1) & (pr01 == 1)))
    fp = int(np.sum((gt01 == 0) & (pr01 == 1)))
    fn = int(np.sum((gt01 == 1) & (pr01 == 0)))
    tn = int(np.sum((gt01 == 0) & (pr01 == 0)))
    return tp, fp, fn, tn

def safe_div(a, b, eps=1e-12):
    return float(a) / float(b + eps)

def cm_metrics(tp, fp, fn, tn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "specificity": float(specificity),
    }


# ----------------------------
# Tuning
# ----------------------------
def sample_config(rng: random.Random):
    cfg = {
        "base_filters": rng.choice([8, 12, 16]),
        "depth": rng.choice([3, 4]),
        "use_bn": rng.choice([True, False]),
        "dropout": rng.choice([0.0, 0.05, 0.10, 0.15]),
        "sep_conv": rng.choice([False, True]),
        "lr": rng.choice([3e-4, 1e-3, 3e-3]),
        "optimizer": rng.choice(["adam", "adamw"]),
        "w_dice": rng.choice([0.6, 0.7, 0.8]),
        "w_bce": rng.choice([0.4, 0.3, 0.2]),
        "use_focal": rng.choice([False, True]),
    }
    # normalisasi bobot loss agar jumlahnya 1 (lebih mudah dijelaskan)
    s = float(cfg["w_dice"] + cfg["w_bce"])
    if s > 0:
        cfg["w_dice"] = float(cfg["w_dice"] / s)
        cfg["w_bce"] = float(cfg["w_bce"] / s)
    return cfg

def tune(args):
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    train_pairs = build_pairs(Path(args.data_root) / "train" / "images", Path(args.data_root) / "train" / "masks")
    val_pairs = build_pairs(Path(args.data_root) / "val" / "images", Path(args.data_root) / "val" / "masks")
    if not train_pairs or not val_pairs:
        raise SystemExit("train/val pairs not found. Check dataset path.")

    best_cfg = None
    best_val_loss = float("inf")

    log_path = out_dir / "severity_tuning_log.csv"
    best_cfg_path = out_dir / "best_severity_config.json"

    # === renamed BEST weights file ===
    best_w_path = out_dir / "best_severity_model.weights.h5"

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "trial",
            "val_loss",
            "base_filters",
            "depth",
            "use_bn",
            "dropout",
            "sep_conv",
            "lr",
            "optimizer",
            "w_dice",
            "w_bce",
            "use_focal",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        rng = random.Random(args.seed)
        for t in range(1, int(args.trials) + 1):
            cfg = sample_config(rng)

            tf.keras.backend.clear_session()
            gc.collect()
            try:
                model = build_unet(cfg, input_shape=(args.img_h, args.img_w, 3))
                model = compile_model(model, cfg)

                tr_ds = make_dataset(train_pairs, args.img_h, args.img_w, args.batch_size, shuffle=True, seed=args.seed + t)
                va_ds = make_dataset(val_pairs, args.img_h, args.img_w, args.batch_size, shuffle=False)

                cb = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=max(1, args.patience // 2),
                        restore_best_weights=True
                    ),
                ]

                hist = model.fit(tr_ds, validation_data=va_ds, epochs=int(args.epochs), verbose=0, callbacks=cb)
                vloss = float(np.min(hist.history.get("val_loss", [np.inf])))

                w.writerow(
                    {
                        "trial": t,
                        "val_loss": vloss,
                        **cfg,
                    }
                )
                f.flush()

                if vloss < best_val_loss:
                    best_val_loss = vloss
                    best_cfg = cfg
                    model.save_weights(str(best_w_path))
                    _save_json(best_cfg, best_cfg_path)

                print(f"[TUNE] trial {t:03d}/{args.trials} | val_loss={vloss:.6f} | best={best_val_loss:.6f}")

            except tf.errors.ResourceExhaustedError:
                print(f"[TUNE] trial {t} OOM -> skipped")
            finally:
                try:
                    del model
                except Exception:
                    pass
                gc.collect()

    print("[DONE] tuning")
    print("[OK] best cfg :", best_cfg_path)
    print("[OK] best wts :", best_w_path)
    print("[OK] log      :", log_path)


# ----------------------------
# Train final
# ----------------------------
def train(args):
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    best_cfg_path = out_dir / "best_severity_config.json"
    if not best_cfg_path.exists():
        raise SystemExit("best_severity_config.json not found. Run tune first (or provide a config).")

    with open(best_cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_pairs = build_pairs(Path(args.data_root) / "train" / "images", Path(args.data_root) / "train" / "masks")
    val_pairs = build_pairs(Path(args.data_root) / "val" / "images", Path(args.data_root) / "val" / "masks")
    if not train_pairs or not val_pairs:
        raise SystemExit("train/val pairs not found. Check dataset path.")

    tf.keras.backend.clear_session()
    gc.collect()

    model = build_unet(cfg, input_shape=(args.img_h, args.img_w, 3))
    model = compile_model(model, cfg)

    tr_ds = make_dataset(train_pairs, args.img_h, args.img_w, args.batch_size, shuffle=True, seed=args.seed)
    va_ds = make_dataset(val_pairs, args.img_h, args.img_w, args.batch_size, shuffle=False)

    # LAST epoch
    final_w_path = out_dir / "final_severity_model.weights.h5"

    # BEST per-epoch checkpoint (monitor val_loss)
    best_w_path = out_dir / "best_severity_model.weights.h5"

    hist_csv = out_dir / "severity_train_history.csv"

    # === callbacks: include ModelCheckpoint to save best-per-epoch and show log ===
    cb = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_w_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, args.patience // 2),
            min_lr=1e-6,
            verbose=1
        ),
    ]

    hist = model.fit(tr_ds, validation_data=va_ds, epochs=int(args.epochs), verbose=1, callbacks=cb)

    model.save_weights(str(final_w_path))
    print("[OK] saved final weights:", final_w_path)
    print("[OK] best weights (checkpoint):", best_w_path)

    keys = list(hist.history.keys())
    with open(hist_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch"] + keys)
        w.writeheader()
        for i in range(len(hist.history[keys[0]])):
            row = {"epoch": i + 1}
            for k in keys:
                row[k] = hist.history[k][i]
            w.writerow(row)

    print("[OK] saved history:", hist_csv)

    # simpan history juga ke JSON (lebih gampang dipakai untuk plotting/paper)
    hist_json = out_dir / "severity_train_history.json"
    hist_clean = {k: [float(x) for x in v] for k, v in hist.history.items()}
    _save_json(hist_clean, hist_json)
    print("[OK] saved history json:", hist_json)

    # plot training curves (loss + dice50 + iou50)
    plot_training_curves(hist_clean, out_dir=out_dir, prefix="severity")


# ----------------------------
# Eval (val set) + outputs
# ----------------------------
def eval_model(args):
    """
    Evaluate model on VAL set:
    - Load cfg ONLY from: models/severity/leaf_mask/best_severity_config.json
    - Load weights from args (--weights) or default best weights in MODEL_DIR
    - Save all eval outputs into RES_DIR (results/...).
    - Report dice/iou thresholded + composite loss (same as training).
    - Optional pixel-level confusion-matrix via --cm_enable
    """
    # ---- output is always RES_DIR ----
    out_dir = Path(RES_DIR)
    ensure_dir(out_dir)

    # ---- cfg must exist (no hardcode fallback) ----
    cfg_path = Path(MODEL_DIR) / "best_severity_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"[ERROR] Config not found (required): {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # ---- weights path ----
    weights_path = Path(args.weights) if args.weights else (Path(MODEL_DIR) / "best_severity_model.weights.h5")
    if not weights_path.exists():
        raise SystemExit(f"[ERROR] Weights not found: {weights_path}")

    # ---- val pairs ----
    val_pairs = build_pairs(
        Path(args.data_root) / "val" / "images",
        Path(args.data_root) / "val" / "masks"
    )
    if not val_pairs:
        raise SystemExit("[ERROR] val pairs not found. Check dataset path.")

    # ---- build model strictly from cfg ----
    tf.keras.backend.clear_session()
    gc.collect()

    model = build_unet(cfg, input_shape=(args.img_h, args.img_w, 3))
    model = compile_model(model, cfg)

    # load weights (should match)
    model.load_weights(str(weights_path))
    print("[OK] loaded weights:", weights_path)
    print("[OK] loaded cfg    :", cfg_path)

    # ---- outputs ----
    eval_dir = out_dir / "eval_outputs"
    pred_mask_dir = eval_dir / "pred_masks"
    masked_dir = eval_dir / "masked_pred"
    ensure_dir(eval_dir)
    ensure_dir(pred_mask_dir)
    ensure_dir(masked_dir)

    # ---- metrics ----
    dice_m = DiceBin(threshold=args.threshold, name="dice50")
    iou_m = IoUBin(threshold=args.threshold, name="iou50")

    # composite loss (same as training)
    loss_sum = 0.0
    loss_cnt = 0

    # optional pixel-level CM aggregation
    agg_tp = agg_fp = agg_fn = agg_tn = 0
    per_image_rows = []

    pbar = tqdm(val_pairs, desc="EVAL", ncols=140)
    for img_path, mask_path in pbar:
        bgr0 = cv2.imread(img_path)
        if bgr0 is None:
            continue

        # resize to eval shape
        bgr = cv2.resize(bgr0, (args.img_w, args.img_h), interpolation=cv2.INTER_LINEAR)
        x = (bgr.astype(np.float32) / 255.0)[None, ...]

        prob = model.predict(x, verbose=0)[0, ..., 0]  # (H,W)
        pred = (prob > args.threshold).astype(np.uint8) * 255

        gt0 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        dice_i = ""
        iou_i = ""
        tp = fp = fn = tn = ""

        if gt0 is not None:
            gt = cv2.resize(gt0, (args.img_w, args.img_h), interpolation=cv2.INTER_NEAREST)
            gt_bin = (gt > 0).astype(np.float32)
            pr_bin = (prob > args.threshold).astype(np.float32)

            # loss per-image (composite)
            yt = tf.convert_to_tensor(gt_bin[None, ..., None], dtype=tf.float32)
            yp = tf.convert_to_tensor(prob[None, ..., None], dtype=tf.float32)

            li = composite_loss(
                yt, yp,
                w_dice=float(cfg["w_dice"]),
                w_bce=float(cfg["w_bce"]),
                use_focal=bool(cfg["use_focal"]),
            )
            loss_sum += float(li.numpy())
            loss_cnt += 1

            # update metrics
            dice_m.update_state(gt_bin[None, ..., None], pr_bin[None, ..., None])
            iou_m.update_state(gt_bin[None, ..., None], pr_bin[None, ..., None])

            # per-image dice/iou (manual, same threshold)
            inter = float((gt_bin * pr_bin).sum())
            denom = float(gt_bin.sum() + pr_bin.sum())
            dice_i = (2.0 * inter + 1.0) / (denom + 1.0 + 1e-9)

            union = float(((gt_bin + pr_bin) > 0).sum())
            iou_i = (inter + 1.0) / (union + 1.0 + 1e-9)

            # optional CM
            if args.cm_enable:
                gt01 = (gt_bin > 0.5).astype(np.uint8)
                pr01 = (pr_bin > 0.5).astype(np.uint8)
                tp, fp, fn, tn = cm_from_binary(gt01, pr01)
                agg_tp += tp
                agg_fp += fp
                agg_fn += fn
                agg_tn += tn

        # per-image row
        row = {
            "filename": Path(img_path).name,
            "dice": float(dice_i) if dice_i != "" else "",
            "iou": float(iou_i) if iou_i != "" else "",
        }
        if args.cm_enable:
            row.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
        per_image_rows.append(row)

        # save pred mask
        stem = Path(img_path).stem
        cv2.imwrite(str(pred_mask_dir / f"{stem}.png"), pred)

        # save masked image back to original size
        m01 = (pred > 0).astype(np.uint8)
        pred_back = cv2.resize(m01, (bgr0.shape[1], bgr0.shape[0]), interpolation=cv2.INTER_NEAREST)
        masked = bgr0 * pred_back[:, :, None]
        cv2.imwrite(str(masked_dir / f"{stem}.jpg"), masked)

    # results
    dice = float(dice_m.result().numpy())
    iou = float(iou_m.result().numpy())
    avg_loss = float(loss_sum / max(loss_cnt, 1))

    weights_stem = Path(weights_path).stem
    thr_tag = f"{float(args.threshold):.2f}".replace(".", "p")

    summary = {
        "weights": str(weights_path),
        "cfg_path": str(cfg_path),
        "threshold": float(args.threshold),
        "loss": float(avg_loss),
        "dice50": float(dice),
        "iou50": float(iou),
        "n_val": int(len(val_pairs)),
    }

    cm_summary = None
    if args.cm_enable:
        cm_summary = cm_metrics(agg_tp, agg_fp, agg_fn, agg_tn)
        summary.update(
            {
                "cm_tp": cm_summary["tp"],
                "cm_fp": cm_summary["fp"],
                "cm_fn": cm_summary["fn"],
                "cm_tn": cm_summary["tn"],
                "cm_precision": cm_summary["precision"],
                "cm_recall": cm_summary["recall"],
                "cm_f1": cm_summary["f1"],
                "cm_accuracy": cm_summary["accuracy"],
                "cm_specificity": cm_summary["specificity"],
            }
        )

    summary_json = eval_dir / f"eval_summary_{weights_stem}_thr{thr_tag}.json"
    summary_csv = eval_dir / f"eval_summary_{weights_stem}_thr{thr_tag}.csv"
    perimg_csv = eval_dir / f"eval_per_image_{weights_stem}_thr{thr_tag}.csv"

    _save_json(summary, summary_json)

    # plot bar metrics (loss/dice/iou)
    plot_eval_metrics_bar(summary, out_dir=eval_dir, prefix=f"eval_metrics_{weights_stem}_thr{thr_tag}")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    per_fields = ["filename", "dice", "iou"]
    if args.cm_enable:
        per_fields += ["tp", "fp", "fn", "tn"]

    with open(perimg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=per_fields)
        w.writeheader()
        for r in per_image_rows:
            w.writerow({k: r.get(k, "") for k in per_fields})

    print("[DONE] eval")
    print("  threshold:", args.threshold)
    print("  loss     :", f"{avg_loss:.6f}")
    print("  dice50   :", f"{dice:.4f}")
    print("  iou50    :", f"{iou:.4f}")
    if args.cm_enable and cm_summary is not None:
        print("  cm(tp,fp,fn,tn):", cm_summary["tp"], cm_summary["fp"], cm_summary["fn"], cm_summary["tn"])
        print("  cm precision   :", f"{cm_summary['precision']:.4f}")
        print("  cm recall      :", f"{cm_summary['recall']:.4f}")
        print("  cm f1          :", f"{cm_summary['f1']:.4f}")
        print("  cm accuracy    :", f"{cm_summary['accuracy']:.4f}")
        print("  cm specificity :", f"{cm_summary['specificity']:.4f}")

    print("  outputs  :", eval_dir)
    print("  summary  :", summary_json)
    print("  per-img  :", perimg_csv)


# ----------------------------
# Predict single image (with GUI picker)
# ----------------------------
def predict_one(args):
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"Weights not found: {weights_path}")

    cfg = {
        "base_filters": 12,
        "depth": 3,
        "use_bn": True,
        "dropout": 0.0,
        "sep_conv": False,
        "lr": 1e-3,
        "optimizer": "adam",
        "w_dice": 0.7,
        "w_bce": 0.3,
        "use_focal": False,
    }

    tf.keras.backend.clear_session()
    gc.collect()
    model = build_unet(cfg, input_shape=(args.img_h, args.img_w, 3))
    model = compile_model(model, cfg)
    model.load_weights(str(weights_path))
    print("[OK] loaded weights:", weights_path)

    img_path = args.image
    if args.gui:
        img_path = pick_image_file_gui()
    if not img_path or not os.path.isfile(img_path):
        raise SystemExit("Image not found (use --image or --gui).")

    bgr0 = cv2.imread(img_path)
    if bgr0 is None:
        raise SystemExit("Failed to read image.")

    bgr = cv2.resize(bgr0, (args.img_w, args.img_h), interpolation=cv2.INTER_LINEAR)
    x = (bgr.astype(np.float32) / 255.0)[None, ...]

    prob = model.predict(x, verbose=0)[0, ..., 0]
    pred = (prob > args.threshold).astype(np.uint8) * 255

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(Path(RES_DIR))

    stem = Path(img_path).stem
    out_mask = out_dir / f"pred_{stem}.png"
    cv2.imwrite(str(out_mask), pred)

    m01 = (pred > 0).astype(np.uint8)
    pred_back = cv2.resize(m01, (bgr0.shape[1], bgr0.shape[0]), interpolation=cv2.INTER_NEAREST)
    masked = bgr0 * pred_back[:, :, None]
    out_masked = out_dir / f"masked_{stem}.jpg"
    cv2.imwrite(str(out_masked), masked)

    print("[DONE] predict")
    print("  image :", img_path)
    print("  thr   :", args.threshold)
    print("  mask  :", out_mask)
    print("  masked:", out_masked)


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser("Severity leaf mask segmentation (binary UNet)")

    sub = p.add_subparsers(dest="mode", required=True)

    def add_common(sp):
        sp.add_argument("--data_root", type=str, default=BASE_DIR)
        sp.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
        sp.add_argument("--img_h", type=int, default=480)
        sp.add_argument("--img_w", type=int, default=640)
        sp.add_argument("--batch_size", type=int, default=1)
        sp.add_argument("--epochs", type=int, default=10)
        sp.add_argument("--patience", type=int, default=6)
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision (if supported).")

    sp_tune = sub.add_parser("tune")
    add_common(sp_tune)
    sp_tune.add_argument("--trials", type=int, default=20)

    sp_train = sub.add_parser("train")
    add_common(sp_train)

    sp_eval = sub.add_parser("eval")
    add_common(sp_eval)
    sp_eval.add_argument("--weights", type=str, default="models/severity/leaf_mask/best_severity_model.weights.h5")
    sp_eval.add_argument("--threshold", type=float, default=0.5)
    sp_eval.add_argument("--cm_enable", action="store_true")

    sp_pred = sub.add_parser("predict")
    add_common(sp_pred)
    sp_pred.add_argument("--weights", type=str, required=True)
    sp_pred.add_argument("--image", type=str, default="")
    sp_pred.add_argument("--threshold", type=float, default=0.5)
    sp_pred.add_argument("--gui", action="store_true")

    return p.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)
    set_memory_growth()
    enable_mixed_precision(bool(args.mixed_precision))

    ensure_dir(Path(args.out_dir))

    if args.mode == "tune":
        tune(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "eval":
        eval_model(args)
    elif args.mode == "predict":
        predict_one(args)
    else:
        raise SystemExit("Unknown mode")

if __name__ == "__main__":
    main()
