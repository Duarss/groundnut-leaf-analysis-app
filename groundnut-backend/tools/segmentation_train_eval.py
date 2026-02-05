# exe_segmentation_model.py
#
# Script ini dipakai untuk:
# (1) TUNING hyperparameter dengan random/grid search (tanpa baseline/anchor).
# (2) TRAIN model terbaik (checkpoint weights) berdasarkan metric tertentu.
# (3) EVAL model (termasuk optional confusion matrix pixel-level + threshold sweep).
# (4) VIZ overlay hasil prediksi vs ground-truth pada val set.
#
# Kenapa "tanpa baseline/anchor"?
# - Agar tuning murni berasal dari search space (random/grid) dan tidak bias dari angka manual.
#
# Model:
# - U-Net decoder + EfficientNetB0 encoder pretrained ImageNet.
# - Output:
#   - per-class: 1 channel (binary mask)
#   - global_4class: 4 channel (mask per kelas)
#
# Dataset:
# - Bisa mix ROI (cropped) dan FULL (uncropped) untuk mengurangi distribution shift.
# - mix_full_ratio:
#   - 0.0 = ROI only
#   - 1.0 = FULL only
#   - (0,1) = mix via sample_from_datasets
# - Kebijakan khusus: ROSETTE dipaksa FULL only (mix_full_ratio=1.0).
#
# Metrics:
# - Dipakai hanya 2 metric biner:
#   - DiceBin@0.5
#   - IoUBin@0.5
#
# Objective tuning:
# - MIN val_loss (terkecil) karena loss adalah target optimisasi yang stabil.
#
# Naming output (disederhanakan untuk sidang):
# - best_{tag}_segmentation_model.weights.h5
# - best_{tag}_segmentation_model_hist.json
# - best_{tag}_segmentation_model_spec.json
# - best_{tag}_tuned_cfg.json
# tag:
# - global => global_4class
# - per-class => singkatan nama kelas (huruf depan tiap kata; 1 kata -> 3 huruf)
#   contoh: ALTERNARIA LEAF SPOT -> als
#
# Catatan penting mengikuti exe_classification_model.py:
# - File cfg hasil tuning disimpan sebagai JSON berisi "pure cfg dict" (tanpa wrapper meta).
# - Audit tuning tetap disimpan sebagai CSV trial (transparansi sidang).
#
# TF 2.10 compatible.

import os, json, argparse, random, itertools, glob, csv, gc, re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageDraw

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose,
    BatchNormalization, Activation, Concatenate, SpatialDropout2D,
    Lambda
)
from tensorflow.keras.applications import EfficientNetB0, efficientnet
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Metric
from tensorflow.keras import mixed_precision


# ========= Performance tweaks =========
# mixed_float16 mempercepat training pada GPU (hemat VRAM), tetapi output layer terakhir dipaksa float32.
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


# ========= Reproducibility =========
# Seed untuk membuat hasil lebih stabil/repeatable (meski tidak 100% deterministik di GPU).
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)


# ========= Paths =========
BASE_DIR  = "utils/segmentation_dataset"
MODEL_DIR = "models/segmentation"
RES_DIR   = "results/segmentation"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

IMG_EXT  = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
MASK_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# ========= Class names (JANGAN SALAH) =========
# Ini harus sama dengan folder class di dataset (untuk global_4class mapping channel).
CLASS_NAMES = [
    "ALTERNARIA LEAF SPOT",
    "LEAF SPOT (EARLY AND LATE)",
    "ROSETTE",
    "RUST",
]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASS_NAMES)}


# ========= Utilities =========
def _safe_name(s: str):
    """Bersihkan string agar aman untuk nama file/folder."""
    s = (s or "").strip()
    if not s:
        return "RUN"
    return "".join([c if c.isalnum() or c in ["_", "-"] else "_" for c in s])

def _slug(s: str):
    """Versi lowercase dari safe_name."""
    return _safe_name(s).lower()

def _save_json(path: str, obj: dict):
    """Simpan dict ke JSON (utf-8)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _load_json(path: str):
    """Load JSON ke dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _cleanup_trial(*objs):
    """Bersihkan objek besar + TF session (mencegah OOM antar trial)."""
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

def _resolve_only_class_case_insensitive(root: str, only_class: str):
    """
    Cari nama class folder secara case-insensitive (mengurangi error CLI saat user mengetik).
    """
    if only_class is None:
        return None
    if not os.path.isdir(root):
        return only_class
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    if not classes:
        return only_class
    want = only_class.strip().lower()
    lut = {c.strip().lower(): c for c in classes}
    if want in lut:
        return lut[want]
    lut2 = {_slug(c): c for c in classes}
    if _slug(only_class) in lut2:
        return lut2[_slug(only_class)]
    return only_class


def _abbr_class_name(class_name: str) -> str:
    """
    Buat singkatan tag yang stabil:
    - Jika >=2 kata: ambil huruf depan tiap kata (ALTERNARIA LEAF SPOT -> als)
    - Jika 1 kata: ambil 3 huruf pertama (ROSETTE -> ros, RUST -> rus)
    Token hanya huruf A-Z (abaikan tanda baca).
    """
    s = (class_name or "").strip().upper()
    tokens = re.findall(r"[A-Z]+", s)
    if not tokens:
        return "cls"
    if len(tokens) >= 2:
        return "".join(t[0] for t in tokens).lower()
    return tokens[0][:3].lower()


def _short_run_tag(args) -> str:
    """Tag pendek untuk naming file output."""
    if args.global_4class:
        return "global"
    return _abbr_class_name(args.only_class)


def _cfg_path_for_run(args) -> str:
    """
    Output cfg hasil tuning (argmin val_loss).
    Contoh: best_als_tuned_cfg.json / best_global_tuned_cfg.json
    """
    tag = _short_run_tag(args)
    return os.path.join(MODEL_DIR, f"best_{tag}_tuned_cfg.json")


def _artifact_paths_for_run(args):
    """
    Nama file artifacts yang ringkas untuk sidang:
    - weights: best_{tag}_segmentation_model.weights.h5
    - hist   : best_{tag}_segmentation_model_hist.json
    - spec   : best_{tag}_segmentation_model_spec.json
    """
    tag = _short_run_tag(args)
    weights = os.path.join(MODEL_DIR, f"best_{tag}_segmentation_model.weights.h5")
    hist    = os.path.join(MODEL_DIR, f"best_{tag}_segmentation_model_hist.json")
    spec    = os.path.join(MODEL_DIR, f"best_{tag}_segmentation_model_spec.json")
    return weights, hist, spec


def _res_root_for_run(args):
    """Folder output overlay/report per run."""
    if args.global_4class:
        return os.path.join(RES_DIR, "global4")
    return os.path.join(RES_DIR, _slug(args.only_class))


def _tuning_trials_csv_path(args, method: str):
    """
    Audit log tuning (semua kandidat + val_loss).
    Dipertahankan untuk bukti bahwa tuning benar-benar random/grid.
    """
    tag = _short_run_tag(args)
    return os.path.join(MODEL_DIR, f"tune_{tag}_{_slug(method)}_trials.csv")


# ============================================================
# NEW: Training curves (loss + dice50 + iou50) and eval bar plot
# ============================================================

def plot_training_curves_seg(history_dict, out_dir, prefix="segmentation"):
    """
    Paper-ready curves untuk segmentation:
    - loss: train loss vs val loss
    - dice50: train dice50 vs val_dice50
    - iou50 : train iou50 vs val_iou50

    Output:
    - <prefix>_curve_loss.png
    - <prefix>_curve_dice.png
    - <prefix>_curve_iou.png
    """
    os.makedirs(out_dir, exist_ok=True)

    if not history_dict or not isinstance(history_dict, dict):
        print("[WARN] Empty/invalid history dict. Skip plotting.")
        return

    # ambil panjang epoch dari key pertama yang list-nya non-empty
    n_epochs = None
    for k, v in history_dict.items():
        if isinstance(v, (list, tuple)) and len(v) > 0:
            n_epochs = len(v)
            break
    if not n_epochs:
        print("[WARN] No epoch data in history. Skip plotting.")
        return

    epochs = np.arange(1, n_epochs + 1)

    def _plot_pair(train_key, val_key, ylabel, title, filename, legend_train, legend_val):
        if train_key not in history_dict:
            print(f"[INFO] Metric not found in history: {train_key}. Skip {filename}")
            return
        plt.figure()
        plt.plot(epochs, history_dict.get(train_key, []), label=legend_train)
        if val_key in history_dict:
            plt.plot(epochs, history_dict.get(val_key, []), label=legend_val)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, filename)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[INFO] Saved plot: {out_path}")

    # LOSS
    _plot_pair(
        train_key="loss",
        val_key="val_loss",
        ylabel="Loss",
        title="Training vs Validation Loss",
        filename=f"{prefix}_curve_loss.png",
        legend_train="train_loss",
        legend_val="val_loss"
    )

    # DICE50
    _plot_pair(
        train_key="dice50",
        val_key="val_dice50",
        ylabel="Dice@0.5",
        title="Training vs Validation Dice@0.5",
        filename=f"{prefix}_curve_dice.png",
        legend_train="train_dice50",
        legend_val="val_dice50"
    )

    # IOU50
    _plot_pair(
        train_key="iou50",
        val_key="val_iou50",
        ylabel="IoU@0.5",
        title="Training vs Validation IoU@0.5",
        filename=f"{prefix}_curve_iou.png",
        legend_train="train_iou50",
        legend_val="val_iou50"
    )


def save_and_plot_eval_metrics(out_clean: dict, out_dir: str, prefix: str):
    """
    Simpan metrics eval ke JSON + plot bar chart (paper-ready).
    Ini memastikan 'loss yang dipakai untuk eval' terdokumentasi jelas di figure.
    """
    os.makedirs(out_dir, exist_ok=True)

    # JSON
    jpath = os.path.join(out_dir, f"{prefix}_metrics.json")
    _save_json(jpath, out_clean)
    print(f"[OK] saved eval metrics json -> {jpath}")

    # Bar chart: prioritaskan loss + dice50 + iou50 (kalau ada)
    keys_pref = ["loss", "dice50", "iou50", "val_loss", "val_dice50", "val_iou50"]
    keys = [k for k in keys_pref if k in out_clean]
    if not keys:
        # fallback: semua key numeric
        keys = [k for k, v in out_clean.items() if isinstance(v, (int, float))]

    vals = [float(out_clean[k]) for k in keys]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(keys)), vals)
    plt.xticks(range(len(keys)), keys, rotation=30, ha="right")
    plt.ylabel("Value")
    plt.title("Evaluation Metrics (Loss/Dice/IoU)")
    plt.tight_layout()
    ppath = os.path.join(out_dir, f"{prefix}_metrics_bar.png")
    plt.savefig(ppath, dpi=200)
    plt.close()
    print(f"[OK] saved eval metrics plot -> {ppath}")


# ========= Constraint helpers =========
def _enforce_loss_sum(cfg: dict, tol: float = 1e-8) -> dict:
    """
    Constraint 1:
    tv_w + bce_w = 1.0
    (bce_w dihitung otomatis dari tv_w)
    """
    c = dict(cfg)
    tv = float(np.clip(float(c.get("tv_w", 0.20)), 0.0, 1.0))
    c["tv_w"] = tv
    c["bce_w"] = float(1.0 - tv)
    s = float(c["tv_w"] + c["bce_w"])
    if abs(s - 1.0) > tol and s > 0:
        c["tv_w"] /= s
        c["bce_w"] /= s
    return c

def _enforce_tversky_ab_sum(cfg: dict, tol: float = 1e-8) -> dict:
    """
    Constraint 2:
    tv_alpha + tv_beta = 1.0
    (dinormalisasi untuk keamanan)
    """
    c = dict(cfg)
    a = float(np.clip(float(c.get("tv_alpha", 0.30)), 0.0, 1.0))
    b = float(np.clip(float(c.get("tv_beta", 0.70)), 0.0, 1.0))
    s = a + b
    if s <= 0:
        a, b, s = 0.50, 0.50, 1.0
    if abs(s - 1.0) > tol:
        a /= s
        b /= s
    c["tv_alpha"] = float(a)
    c["tv_beta"] = float(b)
    return c

def _normalize_and_validate_cfg(c2: dict) -> dict:
    """
    Normalisasi + casting tipe + enforce constraint.
    Ini penting agar training stabil (tidak ada nilai aneh dari grid/random).
    """
    c2 = _enforce_loss_sum(c2)
    c2 = _enforce_tversky_ab_sum(c2)

    c2["mix_full_ratio"] = float(c2["mix_full_ratio"])
    c2["lr"] = float(c2["lr"])
    c2["train_encoder"] = bool(c2["train_encoder"])
    c2["freeze_bn"] = bool(c2["freeze_bn"])
    c2["use_focal"] = bool(c2["use_focal"])
    c2["tv_w"] = float(c2["tv_w"])
    c2["bce_w"] = float(c2["bce_w"])
    c2["tv_alpha"] = float(c2["tv_alpha"])
    c2["tv_beta"] = float(c2["tv_beta"])
    c2["bot_dropout"] = float(c2.get("bot_dropout", 0.0))
    c2["dec_dropout"] = float(c2.get("dec_dropout", 0.0))
    return c2

def _cfg_key(cfg: dict):
    """Key untuk dedup kandidat (mencegah trial identik)."""
    keys = [
        "mix_full_ratio","lr","train_encoder","freeze_bn",
        "tv_w","bce_w","tv_alpha","tv_beta","use_focal",
        "bot_dropout","dec_dropout"
    ]
    return tuple(cfg.get(k) for k in keys)


def _force_rosette_full_only(args, cfg: dict) -> dict:
    """
    Kebijakan khusus ROSETTE:
    - Tidak memakai ROI crop sama sekali.
    - Jadi training harus FULL only => mix_full_ratio = 1.00
    Tujuannya agar mudah dijelaskan saat sidang (aturan jelas, bukan angka random).
    """
    if (not args.global_4class) and (args.only_class == "ROSETTE"):
        c = dict(cfg)
        c["mix_full_ratio"] = 1.0
        return c
    return cfg


def _load_cfg_for_run_or_throw(args) -> dict:
    """
    Karena tidak ada baseline:
    - TRAIN/EVAL/VIZ wajib punya cfg.
    Sumber cfg:
    1) --cfg_json kalau user supply
    2) default: best_{tag}_tuned_cfg.json
    """
    if args.cfg_json:
        if not os.path.exists(args.cfg_json):
            raise RuntimeError(f"--cfg_json not found: {args.cfg_json}")
        raw = _load_json(args.cfg_json)
        cfg = raw.get("selected_cfg", raw)  # support: legacy wrapper (punya selected_cfg) atau pure cfg dict
        return _normalize_and_validate_cfg(cfg)

    cfg_path = _cfg_path_for_run(args)
    if not os.path.exists(cfg_path):
        raise RuntimeError(
            "CFG tidak ditemukan.\n"
            f"- Expected: {cfg_path}\n"
            "- Jalankan dulu: --mode tune (grid/random), atau supply: --cfg_json path.json"
        )
    cached = _load_json(cfg_path)
    cfg = cached.get("selected_cfg", cached)  # support legacy wrapper or pure cfg dict
    return _normalize_and_validate_cfg(cfg)


# ========= Dropout globals =========
# Dropout dipakai di bottleneck dan decoder; diset per cfg agar bisa ikut tuning.
BOT_DROPOUT = 0.0
DEC_DROPOUT = 0.0

def _set_dropout_globals(cfg: dict):
    global BOT_DROPOUT, DEC_DROPOUT
    BOT_DROPOUT = float(cfg.get("bot_dropout", 0.0))
    DEC_DROPOUT = float(cfg.get("dec_dropout", 0.0))


# ================= DATA =================
def _find_mask(mask_dir, stem):
    """Cari file mask berdasarkan stem + daftar ekstensi yang diizinkan."""
    for ext in MASK_EXT:
        mp = os.path.join(mask_dir, stem + ext)
        if os.path.exists(mp):
            return mp
    return None

def _list_classes(root: str):
    """List folder class di root dataset."""
    if not os.path.isdir(root):
        return []
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def collect_pairs(root, keep_class=True, only_class=None):
    """
    Collect (class, image_path, mask_path).
    Struktur dataset:
      root/class_name/images/*.jpg
      root/class_name/masks/*.png
    """
    pairs = []
    if not os.path.isdir(root):
        print(f"[WARN] split not found: {root}")
        return pairs

    classes = _list_classes(root)
    if only_class is not None:
        only_class = _resolve_only_class_case_insensitive(root, only_class)
        if only_class not in classes:
            print(f"[WARN] only_class='{only_class}' not found in {root}. Available: {classes}")
            return []
        classes = [only_class]

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
                pairs.append((cls, ip, mp) if keep_class else (ip, mp))

    msg = f"[DATA] {root}: {len(pairs)} pairs"
    if only_class:
        msg += f" | class={only_class}"
    print(msg)
    return pairs

_WARNED_SWAP = False
def _fix_hw_if_swapped(arr, out_h, out_w, kind="img"):
    """
    Guard kalau ada data yang H/W ketukar.
    Ini sering terjadi kalau ada pipeline yang simpan (W,H) bukan (H,W).
    """
    global _WARNED_SWAP
    if arr.ndim == 3:
        h, w = arr.shape[0], arr.shape[1]
        if (h, w) == (out_h, out_w):
            return arr
        if (h, w) == (out_w, out_h):
            if not _WARNED_SWAP:
                print(f"[WARN] {kind} swapped (got {h}x{w}, expected {out_h}x{out_w}). Auto-transposing once.")
                _WARNED_SWAP = True
            return np.transpose(arr, (1, 0, 2))
        raise TypeError(f"{kind} bad shape {arr.shape}, expected {(out_h,out_w,arr.shape[2])} (or swapped).")
    else:
        h, w = arr.shape[0], arr.shape[1]
        if (h, w) == (out_h, out_w):
            return arr
        if (h, w) == (out_w, out_h):
            if not _WARNED_SWAP:
                print(f"[WARN] {kind} swapped (got {h}x{w}, expected {out_h}x{out_w}). Auto-transposing once.")
                _WARNED_SWAP = True
            return np.transpose(arr, (1, 0))
        raise TypeError(f"{kind} bad shape {arr.shape}, expected {(out_h,out_w)} (or swapped).")

def read_image(p, out_h: int, out_w: int):
    """Load image RGB lalu resize ke ukuran model."""
    img = Image.open(p).convert("RGB")
    img = img.resize((out_w, out_h), Image.BILINEAR)
    arr = np.asarray(img, np.float32) / 255.0
    arr = _fix_hw_if_swapped(arr, out_h, out_w, kind="img")
    return np.clip(arr, 0.0, 1.0).astype(np.float32)

def read_mask_binary(p, out_h: int, out_w: int):
    """Load mask grayscale lalu resize (NEAREST) dan binarisasi."""
    m = Image.open(p).convert("L")
    m = m.resize((out_w, out_h), Image.NEAREST)
    arr = np.asarray(m, np.float32)
    arr = _fix_hw_if_swapped(arr, out_h, out_w, kind="mask")
    arr = (arr > 127).astype(np.float32)
    return np.expand_dims(arr, -1)

def _apply_train_augment_np(x, y, rng: np.random.RandomState, out_h: int, out_w: int):
    """
    Augment ringan & aman untuk segmentation:
    - flip horizontal/vertical
    - rotasi 0/90/180/270
    - sedikit contrast/brightness
    """
    if rng.rand() < 0.5:
        x = x[:, ::-1, :]
        y = y[:, ::-1, :]
    if rng.rand() < 0.2:
        x = x[::-1, :, :]
        y = y[::-1, :, :]

    k = int(rng.randint(0, 4))
    if k:
        x = np.rot90(x, k, axes=(0, 1)).copy()
        y = np.rot90(y, k, axes=(0, 1)).copy()
        # resize ulang untuk jaga konsistensi shape
        if x.shape[0] != out_h or x.shape[1] != out_w:
            x_img = Image.fromarray(np.clip(x * 255.0, 0, 255).astype(np.uint8), mode="RGB")
            x_img = x_img.resize((out_w, out_h), Image.BILINEAR)
            x = np.asarray(x_img, np.float32) / 255.0

            ys = []
            for ci in range(y.shape[-1]):
                y_m = Image.fromarray((y[..., ci] > 0.5).astype(np.uint8) * 255, mode="L")
                y_m = y_m.resize((out_w, out_h), Image.NEAREST)
                yc = (np.asarray(y_m, np.float32) > 127).astype(np.float32)
                ys.append(yc)
            y = np.stack(ys, axis=-1).astype(np.float32)

    if rng.rand() < 0.35:
        c = float(rng.uniform(0.90, 1.10))
        b = float(rng.uniform(-0.06, 0.06))
        x = np.clip(x * c + b, 0.0, 1.0)

    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    y = (y > 0.5).astype(np.float32)
    return x, y

def make_dataset(pairs, img_h, img_w, batch, shuffle=True, augment=False, drop_remainder=False, global_4class=False):
    """
    Buat tf.data Dataset dari list pairs.
    - global_4class:
      y jadi shape (H,W,4) dengan channel sesuai CLASS_TO_INDEX.
    - per-class:
      y jadi shape (H,W,1).
    """
    out_h, out_w = int(img_h), int(img_w)
    n_classes = len(CLASS_NAMES)

    def gen():
        idx = np.arange(len(pairs))
        if shuffle:
            np.random.shuffle(idx)

        # rng lokal agar augment tidak identik antar epoch
        rng = np.random.RandomState(GLOBAL_SEED + int(np.random.randint(0, 10_000_000)))
        for i in idx:
            cls, ip, mp = pairs[i]
            x = read_image(ip, out_h, out_w)
            y_bin = read_mask_binary(mp, out_h, out_w)

            if global_4class:
                if cls not in CLASS_TO_INDEX:
                    continue
                ci = CLASS_TO_INDEX[cls]
                y = np.zeros((out_h, out_w, n_classes), np.float32)
                y[..., ci] = y_bin[..., 0]
            else:
                y = y_bin

            if augment:
                x, y = _apply_train_augment_np(x, y, rng, out_h, out_w)

            yield x, y

    y_shape = (out_h, out_w, len(CLASS_NAMES)) if global_4class else (out_h, out_w, 1)

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

def build_train_dataset_mix(args, cfg):
    """
    MIX ROI + FULL untuk kurangi distribution shift.
    - r<=0   => ROI only
    - r>=1   => FULL only
    - else   => MIX (sample_from_datasets)
    """
    r = float(np.clip(cfg["mix_full_ratio"], 0.0, 1.0))
    roi_root  = os.path.join(BASE_DIR, args.train_roi_split)
    full_root = os.path.join(BASE_DIR, args.train_full_split)

    only = None if args.global_4class else args.only_class
    global4 = bool(args.global_4class)

    # ROI only
    if r <= 0.0:
        tr = collect_pairs(roi_root, keep_class=True, only_class=only)
        if len(tr) == 0:
            raise RuntimeError("Train ROI pairs kosong. Cek train_roi_split & only_class/global_4class.")
        ds = make_dataset(tr, args.img_h, args.img_w, args.batch_size,
                          shuffle=True, augment=True, drop_remainder=True, global_4class=global4)
        info = {"mode": "single_roi", "train_root": roi_root, "train_n": len(tr), "mix_full_ratio": r}
        return ds, info

    # FULL only
    if r >= 1.0:
        tr = collect_pairs(full_root, keep_class=True, only_class=only)
        if len(tr) == 0:
            raise RuntimeError("Train FULL pairs kosong. Cek train_full_split & only_class/global_4class.")
        ds = make_dataset(tr, args.img_h, args.img_w, args.batch_size,
                          shuffle=True, augment=True, drop_remainder=True, global_4class=global4)
        info = {"mode": "single_full", "train_root": full_root, "train_n": len(tr), "mix_full_ratio": r}
        return ds, info

    # MIX
    tr_roi  = collect_pairs(roi_root,  keep_class=True, only_class=only)
    tr_full = collect_pairs(full_root, keep_class=True, only_class=only)
    if len(tr_roi) == 0:
        raise RuntimeError("train_roi_split kosong / tidak ditemukan.")
    if len(tr_full) == 0:
        raise RuntimeError("train_full_split kosong / tidak ditemukan.")

    ds_roi  = make_dataset(tr_roi,  args.img_h, args.img_w, batch=1, shuffle=True, augment=True, drop_remainder=True, global_4class=global4).unbatch()
    ds_full = make_dataset(tr_full, args.img_h, args.img_w, batch=1, shuffle=True, augment=True, drop_remainder=True, global_4class=global4).unbatch()

    ds_mix = tf.data.Dataset.sample_from_datasets([ds_roi, ds_full], weights=[1.0 - r, r], seed=GLOBAL_SEED) \
        .batch(args.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    info = {
        "mode": "mix",
        "roi_root": roi_root, "roi_n": len(tr_roi),
        "full_root": full_root, "full_n": len(tr_full),
        "mix_full_ratio": r
    }
    return ds_mix, info


# ================= METRICS =================
class DiceBin(Metric):
    """
    Dice coefficient setelah thresholding (biner).
    Cocok untuk evaluasi yang "mudah dijelaskan" saat sidang karena benar-benar biner.
    """
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
    """
    IoU (Jaccard) setelah thresholding (biner).
    """
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


# ================= LOSSES =================
def bce_loss(y_true, y_pred):
    """Binary cross-entropy standar."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7), tf.float32)
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

def focal_bce(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal BCE untuk menekan easy negatives (sering membantu kalau mask sparse).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7), tf.float32)
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
    focal_weight = alpha_t * tf.pow(1.0 - p_t, gamma)
    return tf.reduce_mean(focal_weight * bce)

def tversky_loss(y_true, y_pred, alpha=0.70, beta=0.30, smooth=1.0):
    """
    Tversky loss:
    - alpha mengontrol penalti FP
    - beta  mengontrol penalti FN
    Dengan constraint alpha+beta=1 agar interpretasi mudah.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7), tf.float32)
    tp = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=[1,2,3])
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=[1,2,3])
    t = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tf.reduce_mean(t)

def total_loss(y_true, y_pred, cfg):
    """
    Total loss = tv_w * Tversky + bce_w * (BCE atau Focal BCE)
    Dengan constraint: tv_w + bce_w = 1.
    """
    lt = tversky_loss(y_true, y_pred, alpha=cfg["tv_alpha"], beta=cfg["tv_beta"])
    lb = focal_bce(y_true, y_pred) if cfg["use_focal"] else bce_loss(y_true, y_pred)
    return float(cfg["tv_w"]) * lt + float(cfg["bce_w"]) * lb


# ================= MODEL =================
def _conv(x, f, k=3, use_bn=False, drop=0.0):
    """Conv2D -> (BN) -> ReLU -> (SpatialDropout)"""
    x = Conv2D(f, k, padding="same", use_bias=not use_bn)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if drop and drop > 0:
        x = SpatialDropout2D(drop)(x)
    return x

def _safe_concat(x, skip, idx=0):
    """
    Concat skip-connection dengan resize jika bentuk feature map tidak identik.
    Ini mencegah error shape pada beberapa ukuran input.
    """
    def _resize_like(tensors):
        a, b = tensors
        h = tf.shape(a)[1]
        w = tf.shape(a)[2]
        return tf.image.resize(b, (h, w), method="bilinear")
    skip2 = Lambda(_resize_like, name=f"resize_skip_like_{idx}")([x, skip])
    return Concatenate(name=f"concat_{idx}")([x, skip2])

def decoder_block(x, skip, f, drop, idx=0):
    """Upconv -> concat skip -> conv -> conv"""
    x = Conv2DTranspose(f, 2, strides=2, padding="same", name=f"up_{idx}")(x)
    x = _safe_concat(x, skip, idx=idx)
    x = _conv(x, f, k=3, use_bn=False, drop=drop)
    x = _conv(x, f, k=3, use_bn=False, drop=0.0)
    return x

def build_unet_efficientnetb0(net_h, net_w, out_channels=1, train_encoder=False):
    """
    U-Net dengan encoder EfficientNetB0 pretrained ImageNet.
    - train_encoder=False: encoder dibekukan (lebih stabil untuk data kecil).
    - train_encoder=True : fine-tune encoder (bisa lebih akurat tapi rawan overfit).
    """
    inp = Input((net_h, net_w, 3))

    # preprocess_input EfficientNet mengharapkan skala tertentu.
    x0 = Lambda(lambda t: t * 255.0, name="scale_255")(inp)
    x0 = efficientnet.preprocess_input(x0)

    base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x0)
    base.trainable = bool(train_encoder)

    # ambil feature maps untuk skip connections
    s1 = base.get_layer("stem_activation").output
    s2 = base.get_layer("block2a_activation").output
    s3 = base.get_layer("block3a_activation").output
    s4 = base.get_layer("block4a_activation").output
    b  = base.get_layer("top_activation").output

    # bottleneck conv (dropout dituning)
    b = _conv(b, 256, k=3, use_bn=False, drop=BOT_DROPOUT)
    b = _conv(b, 256, k=3, use_bn=False, drop=0.0)

    # decoder (dropout dituning)
    d4 = decoder_block(b,  s4, 256, drop=DEC_DROPOUT, idx=4)
    d3 = decoder_block(d4, s3, 128, drop=DEC_DROPOUT, idx=3)
    d2 = decoder_block(d3, s2, 64,  drop=DEC_DROPOUT, idx=2)
    d1 = decoder_block(d2, s1, 32,  drop=DEC_DROPOUT, idx=1)

    x = Conv2DTranspose(16, 2, strides=2, padding="same", name="up_final")(d1)
    x = _conv(x, 16, k=3, use_bn=False, drop=0.0)

    # output float32 agar stabil walau mixed precision aktif
    out = Conv2D(out_channels, 1, activation="sigmoid", use_bias=True, dtype="float32", name="mask")(x)
    model = Model(inp, out, name="UNet_EfficientNetB0")
    return model, base

def freeze_all_bn(model):
    """
    Freeze BatchNorm:
    - BN sensitif terhadap batch kecil
    - membekukan BN sering membuat fine-tuning lebih stabil (umum dipakai).
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


# ================= OVERLAY UTILS =================
def _to_uint8_rgb(x01):
    """Konversi float [0,1] -> uint8 [0,255]"""
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0 + 0.5).astype(np.uint8)

def _mask_to_rgba(mask01, color=(255, 0, 0), alpha=120):
    """Buat overlay RGBA dari mask biner."""
    m = (mask01 > 0.5).astype(np.uint8) * alpha
    H, W = m.shape[:2]
    r = np.full((H, W), color[0], np.uint8)
    g = np.full((H, W), color[1], np.uint8)
    b = np.full((H, W), color[2], np.uint8)
    a = m.astype(np.uint8)
    return np.stack([r, g, b, a], axis=-1)

def save_val_overlays(model, pairs, img_h, img_w, out_dir, n=24, thr=0.5, seed=42, global_4class=False):
    """
    Simpan overlay pada sebagian val images:
    - GT (hijau)
    - Pred (merah)
    Berguna untuk visualisasi qualitative saat sidang.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    idx = idx[:min(int(n), len(idx))]

    for k, i in enumerate(idx.tolist()):
        cls, ip, mp = pairs[i]
        x = read_image(ip, img_h, img_w)
        y_bin = read_mask_binary(mp, img_h, img_w)[..., 0]

        pred = model.predict(np.expand_dims(x, 0), verbose=0)[0]

        if global_4class:
            if cls not in CLASS_TO_INDEX:
                continue
            ci = CLASS_TO_INDEX[cls]
            pred_ch = pred[..., ci]
            pred_bin = (pred_ch >= float(thr)).astype(np.float32)
            title = f"{cls} | ch={ci}"
        else:
            pred_ch = pred[..., 0]
            pred_bin = (pred_ch >= float(thr)).astype(np.float32)
            title = f"{cls}"

        base = Image.fromarray(_to_uint8_rgb(x), mode="RGB").convert("RGBA")
        gt_rgba = Image.fromarray(_mask_to_rgba(y_bin, color=(0, 255, 0), alpha=110), mode="RGBA")
        pr_rgba = Image.fromarray(_mask_to_rgba(pred_bin, color=(255, 0, 0), alpha=110), mode="RGBA")
        over = Image.alpha_composite(Image.alpha_composite(base, gt_rgba), pr_rgba)

        draw = ImageDraw.Draw(over)
        stem = Path(ip).stem
        draw.text((6, 6), f"{title} | {stem} | thr={thr:.2f}", fill=(255,255,255,255))

        out_path = os.path.join(out_dir, f"val_overlay_{k:03d}_{stem}.png")
        over.convert("RGB").save(out_path)

    print(f"[OK] saved overlays -> {out_dir} (n={len(idx)})")


# ================= TUNING =================
def make_candidates(method, space, tune_trials, ab_tol=1e-6, max_random_tries=200000):
    """
    Generate candidate configs dari search space:
    - grid  : semua kombinasi (dengan filter alpha+beta==1)
    - random: sampling acak (resample jika alpha+beta tidak valid)
    """
    keys = list(space.keys())
    uniq = {}

    def cast(c):
        c2 = dict(c)
        c2["mix_full_ratio"] = float(c2["mix_full_ratio"])
        c2["lr"] = float(c2["lr"])
        c2["train_encoder"] = bool(int(c2["train_encoder"])) if isinstance(c2["train_encoder"], (int, np.integer)) else bool(c2["train_encoder"])
        c2["freeze_bn"]     = bool(int(c2["freeze_bn"]))     if isinstance(c2["freeze_bn"], (int, np.integer))     else bool(c2["freeze_bn"])
        c2["use_focal"]     = bool(int(c2["use_focal"]))     if isinstance(c2["use_focal"], (int, np.integer))     else bool(c2["use_focal"])
        c2["tv_w"] = float(c2["tv_w"])
        c2["tv_alpha"] = float(c2["tv_alpha"])
        c2["tv_beta"]  = float(c2["tv_beta"])
        c2["bot_dropout"] = float(c2.get("bot_dropout", 0.0))
        c2["dec_dropout"] = float(c2.get("dec_dropout", 0.0))
        c2 = _normalize_and_validate_cfg(c2)
        return c2

    if method == "grid":
        raw_count = 1
        for k in keys:
            raw_count *= max(1, len(space[k]))
        if raw_count > 5000:
            print(f"[WARN] Grid cartesian = {raw_count} kombinasi (sebelum filter alpha+beta==1). "
                  "Disarankan pakai --search_method random agar waktu tidak meledak.")

        for combo in itertools.product(*[space[k] for k in keys]):
            raw = dict(zip(keys, combo))
            a0 = float(raw.get("tv_alpha"))
            b0 = float(raw.get("tv_beta"))
            if abs((a0 + b0) - 1.0) > ab_tol:
                continue
            c2 = cast(raw)
            uniq[_cfg_key(c2)] = c2

    elif method == "random":
        domain = {k: list(space[k]) for k in keys}
        target = int(tune_trials)
        tries = 0
        while len(uniq) < target and tries < max_random_tries:
            tries += 1
            raw = {k: random.choice(domain[k]) for k in keys}
            a0 = float(raw.get("tv_alpha"))
            b0 = float(raw.get("tv_beta"))
            if abs((a0 + b0) - 1.0) > ab_tol:
                continue
            c2 = cast(raw)
            uniq[_cfg_key(c2)] = c2

        if len(uniq) < target:
            print(f"[WARN] valid random candidates hanya {len(uniq)}/{target}. "
                  f"Pastikan tv_alpha & tv_beta berisi pasangan komplementer (mis 0.3 dan 0.7).")
    else:
        raise ValueError("method must be 'grid' or 'random'")

    return list(uniq.values())

def run_one_trial(val_pairs, args, cfg, tune_epochs, tune_patience):
    """
    Jalankan 1 trial proxy:
    - training singkat dengan early stopping
    - objective: min val_loss
    """
    tf.keras.backend.clear_session()

    cfg = _normalize_and_validate_cfg(cfg)
    cfg = _force_rosette_full_only(args, cfg)

    _set_dropout_globals(cfg)

    net_h, net_w = int(args.img_h), int(args.img_w)
    out_ch = len(CLASS_NAMES) if args.global_4class else 1

    # tuning bs bisa lebih kecil untuk anti-OOM
    tune_bs = int(args.tune_batch_size)

    old_bs = int(args.batch_size)
    args.batch_size = tune_bs
    try:
        ds_tr, _ = build_train_dataset_mix(args, cfg)
    finally:
        args.batch_size = old_bs

    ds_va = make_dataset(val_pairs, net_h, net_w, tune_bs, shuffle=False, augment=False, global_4class=args.global_4class)

    model, _ = build_unet_efficientnetb0(net_h, net_w, out_channels=out_ch, train_encoder=bool(cfg["train_encoder"]))
    model.build((None, net_h, net_w, 3))

    if cfg["freeze_bn"]:
        freeze_all_bn(model)

    loss_fn = lambda yt, yp: total_loss(yt, yp, cfg)

    # metric untuk monitoring tambahan (bukan objective tuning)
    metrics = [
        DiceBin(thr=0.50, name="dice50"),
        IoUBin(thr=0.50, name="iou50"),
    ]

    opt = tf.keras.optimizers.Adam(learning_rate=float(cfg["lr"]), clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=int(tune_patience),
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=0
    )

    fit_kwargs = dict(epochs=int(tune_epochs), callbacks=[es], verbose=0)
    if int(args.tune_val_steps) > 0:
        fit_kwargs["validation_steps"] = int(args.tune_val_steps)

    model.fit(ds_tr, validation_data=ds_va, **fit_kwargs)

    hist = model.history.history
    best_val_loss = float(np.min(hist.get("val_loss", [np.inf])))
    best_val_dice = float(np.max(hist.get("val_dice50", [-1.0])))

    _cleanup_trial(model, opt, ds_tr, ds_va)
    return best_val_loss, best_val_dice

def tuning_search(args, method, tune_trials, tune_epochs, tune_patience, space):
    """
    Proses tuning lengkap:
    - Generate kandidat
    - Evaluate tiap kandidat (run_one_trial)
    - Ambil best_cfg = argmin(val_loss)
    - Simpan:
      * trials csv (audit)
      * best tuned cfg json (pure dict, dipakai train/eval/viz)
    """
    val_root = os.path.join(BASE_DIR, args.val_split)
    va = collect_pairs(val_root, keep_class=True, only_class=None if args.global_4class else args.only_class)
    if len(va) == 0:
        raise RuntimeError("Val pairs kosong. Cek val_split & only_class/global_4class.")

    trials_csv = _tuning_trials_csv_path(args, method)
    cfg_out_path = _cfg_path_for_run(args)

    candidates = make_candidates(method, space, tune_trials)

    # Deduplikasi + enforce ROSETTE policy
    uniq = {}
    for c in candidates:
        c2 = _normalize_and_validate_cfg(c)
        c2 = _force_rosette_full_only(args, c2)
        uniq[_cfg_key(c2)] = c2
    candidates = list(uniq.values())

    if len(candidates) == 0:
        raise RuntimeError("Tidak ada candidate valid setelah constraint filter (alpha+beta=1).")

    print(f"\n🔎 Tuning {method}: {len(candidates)} candidate(s) | objective=MIN val_loss")
    print(f"[TUNE] tune_batch_size={args.tune_batch_size} | tune_val_steps={args.tune_val_steps}")
    print(f"[TUNE] output cfg -> {cfg_out_path}")

    results = []
    for i, cfg in enumerate(candidates, start=1):
        print(f"  Trial {i}/{len(candidates)} -> {cfg}")
        vloss, vdice = run_one_trial(va, args, cfg, tune_epochs, tune_patience)
        results.append({**cfg, "best_val_loss": vloss, "best_val_dice": vdice})

    results_sorted = sorted(results, key=lambda r: float(r["best_val_loss"]))

    # Simpan audit csv
    with open(trials_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results_sorted[0].keys()))
        w.writeheader()
        for r in results_sorted:
            w.writerow(r)
    print(f"📝 Saved tuning trials -> {trials_csv}")

    # best cfg = argmin val_loss
    best = results_sorted[0]
    best_cfg = {k: best[k] for k in [
        "mix_full_ratio","lr","train_encoder","freeze_bn",
        "tv_w","bce_w","tv_alpha","tv_beta","use_focal",
        "bot_dropout","dec_dropout"
    ]}
    best_cfg = _normalize_and_validate_cfg(best_cfg)
    best_cfg = _force_rosette_full_only(args, best_cfg)

    # Simpan hanya cfg (tanpa wrapper) agar sederhana seperti exe_classification_model.py.
    # Audit trail tetap ada di trials CSV.
    _save_json(cfg_out_path, dict(best_cfg))
    print(f"🏆 Saved best tuned cfg -> {cfg_out_path}")

    return dict(best_cfg)


# ================= TRAIN / EVAL / VIZ =================
def _pixel_confusion_from_probs(y_true01: np.ndarray, y_prob01: np.ndarray, thr: float = 0.5):
    """
    Confusion matrix pixel-level:
    - dipakai untuk menghitung TP/FP/FN/TN berdasarkan threshold.
    """
    yt = (y_true01 >= 0.5).astype(np.uint8).reshape(-1)
    yp = (y_prob01 >= float(thr)).astype(np.uint8).reshape(-1)

    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return tn, fp, fn, tp

def _cm_report_from_counts(tn: int, fp: int, fn: int, tp: int):
    """
    Report sederhana (precision/recall/f1/accuracy/specificity/iou/dice).
    Ini untuk melengkapi evaluasi saat sidang.
    """
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    specificity = tn / (tn + fp + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    support_pos = tp + fn
    support_neg = tn + fp
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "specificity": float(specificity),
        "iou": float(iou),
        "dice": float(dice),
        "support_pos": int(support_pos),
        "support_neg": int(support_neg),
    }


# ================= TRAIN / EVAL / VIZ =================
def _to_uint8_gray(mask01):
    return (np.clip(mask01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

def _run_tag(args, cfg):
    """
    Tag panjang untuk folder overlay (opsional), bukan untuk file utama (file utama pakai best_{tag}_...).
    """
    run_tag = args.tag.strip() if args.tag.strip() else ("GLOBAL4" if args.global_4class else args.only_class)
    run_tag = _safe_name(run_tag)

    net_h, net_w = int(args.img_h), int(args.img_w)
    mix_tag = f"mix{int(cfg['mix_full_ratio']*100):02d}" if cfg["mix_full_ratio"] > 0 else "single"
    enc_tag = "encON" if cfg["train_encoder"] else "encOFF"
    bn_tag  = "bnFZ" if cfg["freeze_bn"] else "bnTR"
    focal_tag = "focal" if cfg["use_focal"] else "bce"

    tag = (
        f"unet_effb0_in{net_h}x{net_w}_"
        f"{_slug(run_tag)}_{mix_tag}_{enc_tag}_{bn_tag}_"
        f"lr{cfg['lr']:.0e}_tv{cfg['tv_w']:.2f}_bce{cfg['bce_w']:.2f}_"
        f"a{cfg['tv_alpha']:.2f}_b{cfg['tv_beta']:.2f}_{focal_tag}_"
        f"bot{cfg['bot_dropout']:.2f}_dec{cfg['dec_dropout']:.2f}"
    ).replace(".", "p").replace("+", "")
    return tag

def tune_mode(args):
    """
    Mode tuning:
    - menghasilkan best_{tag}_tuned_cfg.json (pure dict)
    """
    if args.search_method == "none":
        raise RuntimeError("Tanpa baseline, --search_method none tidak didukung. Pilih: grid atau random.")

    # ROSETTE dipaksa FULL only
    grid_mix = args.grid_mix
    if (not args.global_4class) and (args.only_class == "ROSETTE"):
        grid_mix = [1.0]
        print("[INFO] ROSETTE policy: forcing mix_full_ratio=1.00 (FULL only). grid_mix overridden to [1.0].")

    space = {
        "lr": args.grid_lr,
        "mix_full_ratio": grid_mix,
        "train_encoder": args.grid_train_encoder,
        "freeze_bn": args.grid_freeze_bn,
        "tv_w": args.grid_tv_w,
        "tv_alpha": args.grid_tv_alpha,
        "tv_beta": args.grid_tv_beta,
        "use_focal": args.grid_use_focal,
        "bot_dropout": args.grid_bot_dropout,
        "dec_dropout": args.grid_dec_dropout,
    }

    tuning_search(
        args=args,
        method=args.search_method,
        tune_trials=args.tune_trials,
        tune_epochs=args.tune_epochs,
        tune_patience=args.tune_patience,
        space=space
    )

def train_mode(args):
    """
    Mode training:
    - load cfg (wajib ada)
    - training penuh (epochs besar)
    - simpan weights terbaik + history + spec
    - NEW: plot training curves (loss + dice + iou)
    """
    cfg = _load_cfg_for_run_or_throw(args)
    cfg = _force_rosette_full_only(args, cfg)
    _set_dropout_globals(cfg)

    ds_tr, info = build_train_dataset_mix(args, cfg)

    va_root = os.path.join(BASE_DIR, args.val_split)
    va = collect_pairs(va_root, keep_class=True, only_class=None if args.global_4class else args.only_class)
    if len(va) == 0:
        raise RuntimeError("Val pairs kosong. Cek val_split & only_class/global_4class.")
    ds_va = make_dataset(va, args.img_h, args.img_w, args.batch_size, shuffle=False, augment=False, global_4class=args.global_4class)

    net_h, net_w = int(args.img_h), int(args.img_w)
    out_ch = len(CLASS_NAMES) if args.global_4class else 1

    model, _ = build_unet_efficientnetb0(net_h, net_w, out_channels=out_ch, train_encoder=bool(cfg["train_encoder"]))
    model.build((None, net_h, net_w, 3))

    # freeze BN untuk stabilitas fine-tuning
    if cfg["freeze_bn"]:
        freeze_all_bn(model)

    loss_fn = lambda yt, yp: total_loss(yt, yp, cfg)

    metrics = [
        DiceBin(thr=0.50, name="dice50"),
        IoUBin(thr=0.50, name="iou50"),
    ]

    weights_path, hist_path, spec_path = _artifact_paths_for_run(args)

    # monitor metric untuk checkpointing:
    # - kalau val_loss => mode 'min'
    # - selain itu => mode 'max'
    monitor = args.monitor_metric
    mode = "max" if monitor != "val_loss" else "min"

    ckpt_best = ModelCheckpoint(weights_path, monitor=monitor, mode=mode, save_best_only=True, save_weights_only=True, verbose=1)
    es = EarlyStopping(monitor=monitor, mode=mode, patience=args.es_patience, restore_best_weights=True, verbose=1)
    rlr = ReduceLROnPlateau(monitor=monitor, mode=mode, factor=0.5, patience=args.lr_patience, min_lr=args.min_lr, verbose=1)

    print("\n[TRAIN_SETUP]")
    print(f"  run            : {'GLOBAL4' if args.global_4class else args.only_class}")
    print(f"  data_mode      : {info.get('mode')}")
    print(f"  input          : {net_h}x{net_w} | bs={args.batch_size}")
    print(f"  cfg            : {cfg}")
    print(f"  monitor        : {monitor} ({mode})")
    print(f"  ckpt(best)     : {weights_path}")

    opt = tf.keras.optimizers.Adam(learning_rate=float(cfg["lr"]), clipnorm=1.0)
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

    hist = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, callbacks=[ckpt_best, es, rlr], verbose=1)

    # Simpan history training (loss + metrics per epoch)
    hist_clean = {k: [float(x) for x in v] for k, v in hist.history.items()}
    _save_json(hist_path, hist_clean)
    print(f"[OK] saved history -> {hist_path}")

    # NEW: plot training curves (loss + dice + iou)
    tag = _short_run_tag(args)
    prefix = f"best_{tag}_segmentation_model"
    plot_training_curves_seg(hist_clean, out_dir=MODEL_DIR, prefix=prefix)

    # Simpan spec: memudahkan audit/repro saat sidang
    # Catatan: best cfg disimpan sebagai pure dict (best_{tag}_tuned_cfg.json).
    # Audit tuning tersimpan di CSV: tune_{tag}_{method}_trials.csv.
    spec = {
        "model": "UNet_EfficientNetB0",
        "img_size": [int(args.img_h), int(args.img_w)],
        "run_type": "global4" if args.global_4class else "per_class",
        "only_class": None if args.global_4class else args.only_class,
        "class_names": CLASS_NAMES,
        "selected_cfg": cfg,
        "cfg_source": args.cfg_json if args.cfg_json else _cfg_path_for_run(args),
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
    _save_json(spec_path, spec)

    # Optional: simpan overlay agar bisa ditunjukkan saat sidang
    if args.save_overlays:
        res_root = _res_root_for_run(args)
        folder_tag = _safe_name(args.tag.strip() if args.tag.strip() else ("GLOBAL4" if args.global_4class else args.only_class))
        out_dir = os.path.join(res_root, f"{_slug(folder_tag)}_overlays_thr{args.overlay_thr:.2f}".replace(".", "p"))
        save_val_overlays(model, va, net_h, net_w, out_dir, n=args.overlay_n, thr=args.overlay_thr,
                          seed=GLOBAL_SEED, global_4class=args.global_4class)

    _cleanup_trial(model, opt, ds_tr, ds_va)

def eval_mode(args):
    """
    Mode evaluasi:
    - load cfg (wajib) agar loss yang dipakai sama dengan training/tuning
    - evaluate metrics
    - NEW:
      * simpan eval metrics json + bar plot (loss/dice/iou)
    - optional:
      * cm_enable: confusion matrix pixel-level
      * thr_sweep: cari threshold terbaik untuk dice
    """
    tf.keras.backend.clear_session()

    val_root = os.path.join(BASE_DIR, args.val_split)
    va = collect_pairs(val_root, keep_class=True, only_class=None if args.global_4class else args.only_class)
    if len(va) == 0:
        raise RuntimeError("Val pairs kosong. Cek val_split & only_class/global_4class.")

    net_h, net_w = int(args.img_h), int(args.img_w)
    out_ch = len(CLASS_NAMES) if args.global_4class else 1

    # cfg wajib ada (tanpa baseline)
    cfg = _load_cfg_for_run_or_throw(args)
    cfg = _force_rosette_full_only(args, cfg)
    _set_dropout_globals(cfg)

    model, _ = build_unet_efficientnetb0(net_h, net_w, out_channels=out_ch, train_encoder=False)
    model.build((None, net_h, net_w, 3))
    model.load_weights(args.weights)

    if cfg["freeze_bn"] or args.freeze_bn:
        freeze_all_bn(model)

    ds = make_dataset(va, net_h, net_w, args.batch_size, shuffle=False, augment=False, global_4class=args.global_4class)

    # IMPORTANT: loss eval memakai total_loss yang sama seperti training/tuning
    loss_fn = lambda yt, yp: total_loss(yt, yp, cfg)
    metrics = [
        DiceBin(thr=0.50, name="dice50"),
        IoUBin(thr=0.50, name="iou50"),
    ]

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_fn, metrics=metrics)

    out = model.evaluate(ds, verbose=1, return_dict=True)
    out_clean = {k: float(v) for k, v in out.items()}
    print("[EVAL]", json.dumps(out_clean, indent=2))

    # NEW: save eval metrics json + plot
    res_root = _res_root_for_run(args)
    os.makedirs(res_root, exist_ok=True)
    tag = _slug("global4" if args.global_4class else args.only_class)
    prefix = f"eval_{tag}_{Path(args.weights).stem}"
    save_and_plot_eval_metrics(out_clean, out_dir=res_root, prefix=prefix)

    # ===== Confusion matrix + report (pixel-level) =====
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
        if args.global_4class:
            for ci, cname in enumerate(CLASS_NAMES):
                tn, fp, fn, tp = _pixel_confusion_from_probs(y_true[..., ci], y_prob[..., ci], thr=thr_cm)
                rep = _cm_report_from_counts(tn, fp, fn, tp)
                rep["class"] = cname
                reports.append(rep)
                print(f"  {cname}: TP={tp} FP={fp} FN={fn} TN={tn} | P={rep['precision']:.4f} R={rep['recall']:.4f} F1={rep['f1']:.4f}")

            macro = {
                "class": "MACRO_AVG",
                "precision": float(np.mean([r["precision"] for r in reports])),
                "recall": float(np.mean([r["recall"] for r in reports])),
                "f1": float(np.mean([r["f1"] for r in reports])),
                "iou": float(np.mean([r["iou"] for r in reports])),
                "dice": float(np.mean([r["dice"] for r in reports])),
            }
            reports.append(macro)
        else:
            tn, fp, fn, tp = _pixel_confusion_from_probs(y_true[..., 0], y_prob[..., 0], thr=thr_cm)
            rep = _cm_report_from_counts(tn, fp, fn, tp)
            rep["class"] = args.only_class
            reports.append(rep)
            print(f"  {args.only_class}: TP={tp} FP={fp} FN={fn} TN={tn} | P={rep['precision']:.4f} R={rep['recall']:.4f} F1={rep['f1']:.4f}")

        cm_tag = _slug('global4' if args.global_4class else args.only_class)
        cm_base = f"cm_report_{cm_tag}_{Path(args.weights).stem}_thr{thr_cm:.2f}".replace(".", "p")

        cm_json = os.path.join(res_root, f"{cm_base}.json")
        _save_json(cm_json, {"thr": thr_cm, "global_4class": bool(args.global_4class), "reports": reports})
        cm_csv  = os.path.join(res_root, f"{cm_base}.csv")

        cols = ["class","tp","fp","fn","tn","precision","recall","f1","accuracy","specificity","iou","dice","support_pos","support_neg"]
        with open(cm_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in reports:
                w.writerow({c: r.get(c, "") for c in cols})

        print(f"[OK] saved cm report -> {cm_json}")
        print(f"[OK] saved cm report -> {cm_csv}")

    # ===== Threshold sweep (optional) =====
    if args.thr_sweep:
        print("\n[THRESHOLD_SWEEP] searching best binary dice on VAL...")
        thrs = np.linspace(float(args.thr_min), float(args.thr_max), int(args.thr_steps)).tolist()

        ys, ps = [], []
        for xb, yb in ds:
            pb = model.predict(xb, verbose=0)
            ys.append(yb.numpy())
            ps.append(pb)
        y_true = np.concatenate(ys, axis=0)
        y_pred = np.concatenate(ps, axis=0)

        rows = []

        if args.global_4class:
            for ci, cname in enumerate(CLASS_NAMES):
                yt = y_true[..., ci]
                yp = y_pred[..., ci]
                best = (-1.0, None)
                for t in thrs:
                    pr = (yp >= float(t)).astype(np.float32)
                    inter = (yt * pr).sum(axis=(1,2))
                    union = (yt + pr).sum(axis=(1,2))
                    dice = (2.0 * inter + 1.0) / (union + 1.0 + 1e-9)
                    md = float(dice.mean())
                    rows.append({"class": cname, "thr": float(t), "dice_bin": md})
                    if md > best[0]:
                        best = (md, float(t))
                print(f"[BEST] {cname}: thr={best[1]:.3f} dice_bin={best[0]:.4f}")
        else:
            yt = y_true[..., 0]
            yp = y_pred[..., 0]
            best = (-1.0, None)
            for t in thrs:
                pr = (yp >= float(t)).astype(np.float32)
                inter = (yt * pr).sum(axis=(1,2))
                union = (yt + pr).sum(axis=(1,2))
                dice = (2.0 * inter + 1.0) / (union + 1.0 + 1e-9)
                md = float(dice.mean())
                rows.append({"class": args.only_class, "thr": float(t), "dice_bin": md})
                if md > best[0]:
                    best = (md, float(t))
            print(f"[BEST] thr={best[1]:.3f} dice_bin={best[0]:.4f}")

        out_csv = os.path.join(res_root, f"thr_sweep_{_slug('global4' if args.global_4class else args.only_class)}_{Path(args.weights).stem}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["class","thr","dice_bin"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[OK] saved thr sweep -> {out_csv}")

    # overlay (optional)
    if args.save_overlays:
        out_dir = os.path.join(res_root, f"eval_{Path(args.weights).stem}_overlays_thr{args.overlay_thr:.2f}".replace(".", "p"))
        save_val_overlays(model, va, net_h, net_w, out_dir, n=args.overlay_n, thr=args.overlay_thr,
                          seed=GLOBAL_SEED, global_4class=args.global_4class)

    _cleanup_trial(model, ds)

def viz_mode(args):
    """
    Mode visualisasi overlay saja (tanpa hitung metric).
    """
    tf.keras.backend.clear_session()

    val_root = os.path.join(BASE_DIR, args.val_split)
    va = collect_pairs(val_root, keep_class=True, only_class=None if args.global_4class else args.only_class)
    if len(va) == 0:
        raise RuntimeError("Val pairs kosong. Cek val_split & only_class/global_4class.")

    net_h, net_w = int(args.img_h), int(args.img_w)
    out_ch = len(CLASS_NAMES) if args.global_4class else 1

    cfg = _load_cfg_for_run_or_throw(args)
    cfg = _force_rosette_full_only(args, cfg)
    _set_dropout_globals(cfg)

    model, _ = build_unet_efficientnetb0(net_h, net_w, out_channels=out_ch, train_encoder=False)
    model.build((None, net_h, net_w, 3))
    model.load_weights(args.weights)

    if args.freeze_bn:
        freeze_all_bn(model)

    tag = args.tag.strip() if args.tag.strip() else Path(args.weights).stem
    res_root = _res_root_for_run(args)
    out_dir = os.path.join(res_root, f"{_slug(tag)}_val_overlays_thr{args.thr:.2f}".replace(".", "p"))
    save_val_overlays(model, va, net_h, net_w, out_dir, n=args.n, thr=args.thr,
                      seed=GLOBAL_SEED, global_4class=args.global_4class)

    _cleanup_trial(model)


# ================= CLI =================
def parse_args():
    p = argparse.ArgumentParser(description="Tune/Train/Eval/Viz U-Net EffNetB0 segmentation (anti-OOM).")

    p.add_argument("--mode", choices=["tune", "train", "eval", "viz"], default="train")

    p.add_argument("--global_4class", action="store_true")
    p.add_argument("--only_class", default=None)

    # opsional: pakai cfg json tertentu (mis. untuk reproduce run)
    p.add_argument("--cfg_json", default="",
                   help="Path cfg json. Kalau kosong, default pakai best_{tag}_tuned_cfg.json.")

    p.add_argument("--img_h", type=int, default=480)
    p.add_argument("--img_w", type=int, default=640)

    # GPU 4GB: default lebih aman
    p.add_argument("--batch_size", type=int, default=2)

    p.add_argument("--train_roi_split",  default="train_roi")
    p.add_argument("--train_full_split", default="train_balanced_perclass")
    p.add_argument("--val_split", default="val")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--lr_patience", type=int, default=6)
    p.add_argument("--es_patience", type=int, default=16)

    # monitoring metric (train)
    p.add_argument("--monitor_metric", default="val_dice50",
                   choices=["val_loss", "val_dice50", "val_iou50"],
                   help="Metric untuk ckpt/ES/RLR. Default val_dice50.")

    # weights (eval/viz)
    p.add_argument("--weights", default="")

    # overlays
    p.add_argument("--save_overlays", action="store_true")
    p.add_argument("--overlay_n", type=int, default=30)
    p.add_argument("--overlay_thr", type=float, default=0.50)
    p.add_argument("--tag", default="")

    # viz
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--thr", type=float, default=0.50)

    # eval extras
    p.add_argument("--thr_sweep", action="store_true")
    p.add_argument("--thr_min", type=float, default=0.10)
    p.add_argument("--thr_max", type=float, default=0.90)
    p.add_argument("--thr_steps", type=int, default=17)

    # confusion-matrix style report (pixel-level) for eval
    p.add_argument("--cm_enable", action="store_true",
                   help="Jika diaktifkan, eval_mode akan menghitung confusion matrix pixel-level dan report.")
    p.add_argument("--cm_thr", type=float, default=0.50,
                   help="Threshold untuk confusion matrix pixel-level.")

    # tuning controls (anti-OOM)
    p.add_argument("--search_method", choices=["grid", "random", "none"], default="random")
    p.add_argument("--tune_trials", type=int, default=50)
    p.add_argument("--tune_epochs", type=int, default=15)
    p.add_argument("--tune_patience", type=int, default=6)
    p.add_argument("--tune_batch_size", type=int, default=1,
                   help="Batch size ONLY untuk tuning trials (anti-OOM).")
    p.add_argument("--tune_val_steps", type=int, default=0,
                   help="Jika >0, batasi validation_steps saat tuning untuk lebih ringan.")

    # search space (diskret)
    p.add_argument("--grid_lr", type=float, nargs="+",
                   default=[5e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4])
    p.add_argument("--grid_mix", type=float, nargs="+",
                   default=[0.50, 0.60, 0.70, 0.80, 0.90, 1.00])

    p.add_argument("--grid_train_encoder", type=int, nargs="+", default=[0, 1])
    p.add_argument("--grid_freeze_bn", type=int, nargs="+", default=[1])

    # tv_w akan menentukan bce_w otomatis (bce_w = 1-tv_w)
    p.add_argument("--grid_tv_w", type=float, nargs="+",
                   default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])

    # alpha+beta harus 1 -> grid akan difilter, random akan di-resample
    p.add_argument("--grid_tv_alpha", type=float, nargs="+",
                   default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    p.add_argument("--grid_tv_beta", type=float, nargs="+",
                   default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])

    p.add_argument("--grid_use_focal", type=int, nargs="+", default=[0, 1])

    p.add_argument("--grid_bot_dropout", type=float, nargs="+",
                   default=[0.00, 0.03, 0.06, 0.10, 0.14])
    p.add_argument("--grid_dec_dropout", type=float, nargs="+",
                   default=[0.00, 0.02, 0.04, 0.06, 0.08, 0.10])

    # eval override (opsional) - tetap dipertahankan minimal (BN)
    p.add_argument("--freeze_bn", action="store_true")

    args = p.parse_args()

    if args.global_4class:
        args.only_class = None

    if (not args.global_4class) and (args.only_class is None):
        raise RuntimeError("Per-class mode butuh --only_class. Atau pakai --global_4class.")

    if args.mode in ["eval", "viz"] and not args.weights:
        raise RuntimeError("--weights wajib untuk mode eval/viz")

    return args


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "tune":
        tune_mode(args)
    elif args.mode == "train":
        train_mode(args)
    elif args.mode == "eval":
        eval_mode(args)
    else:
        viz_mode(args)
