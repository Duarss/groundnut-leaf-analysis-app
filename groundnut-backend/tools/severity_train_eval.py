# tools/severity_train_eval.py
import os, json, random, argparse, csv, gc, cv2, re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.metrics import Metric
from tqdm import tqdm

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# ========= Paths =========
BASE_DIR  = "datasets/processed/severity_dataset"
MODEL_DIR = "models/severity/leaf_mask"
RES_DIR   = "datasets/results/severity/leaf_mask"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

DEFAULT_OUT_DIR = MODEL_DIR
GLOBAL_SEED = 42

# ======================
# Setup / Utils
# ======================
def seed_everything(seed=GLOBAL_SEED):
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
        mixed_precision.set_global_policy("mixed_float16" if enable else "float32")
    except Exception:
        pass

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

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

def _to_builtin(obj):
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def save_json(obj, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_to_builtin)

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def normalize_cfg(cfg: dict) -> dict:
    c = dict(cfg)

    wd = float(c.get("w_dice", 0.7))
    wb = float(c.get("w_bce", 0.3))
    s = wd + wb
    if s <= 0:
        wd, wb, s = 0.7, 0.3, 1.0
    c["w_dice"] = float(wd / s)
    c["w_bce"] = float(wb / s)

    c["base_filters"] = int(c.get("base_filters", 12))
    c["depth"] = int(c.get("depth", 3))
    c["use_bn"] = bool(c.get("use_bn", True))
    c["dropout"] = float(c.get("dropout", 0.0))
    c["sep_conv"] = bool(c.get("sep_conv", False))
    c["lr"] = float(c.get("lr", 1e-3))
    c["optimizer"] = str(c.get("optimizer", "adam")).lower()
    c["use_focal"] = bool(c.get("use_focal", False))
    return c

# ======================
# Plotting
# ======================
def plot_training_curves(history_dict, out_dir, prefix="severity"):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    if not history_dict or not isinstance(history_dict, dict):
        print("[WARN] history kosong. Skip plotting.")
        return

    n_epochs = None
    for _, v in history_dict.items():
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

# ======================
# IO helpers
# ======================
def list_images(img_dir: Path):
    out = []
    for p in sorted(img_dir.glob("*")):
        if p.suffix.lower() in VALID_EXT:
            out.append(str(p))
    return out

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

# ======================
# Dataset pipeline
# ======================
def decode_image(path, out_h, out_w):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return bgr.astype(np.float32) / 255.0

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

def make_dataset(pairs, out_h, out_w, batch_size, shuffle=False, seed=GLOBAL_SEED):
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
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ======================
# Losses
# ======================
def soft_dice_for_loss(y_true, y_pred, eps=1e-7):
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

# ======================
# Model
# ======================
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
    cfg = normalize_cfg(cfg)
    base_filters = int(cfg["base_filters"])
    depth = int(cfg["depth"])
    use_bn = bool(cfg["use_bn"])
    dropout = float(cfg["dropout"])
    sep_conv = bool(cfg["sep_conv"])

    inp = tf.keras.Input(shape=input_shape)

    skips = []
    x = inp
    f = base_filters
    for _ in range(depth):
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
    cfg = normalize_cfg(cfg)
    lr = float(cfg["lr"])
    opt_name = cfg["optimizer"]

    if opt_name == "adamw":
        opt = tf.keras.optimizers.AdamW(lr)
    else:
        opt = tf.keras.optimizers.Adam(lr)

    model.compile(
        optimizer=opt,
        loss=lambda yt, yp: composite_loss(
            yt, yp,
            w_dice=float(cfg["w_dice"]),
            w_bce=float(cfg["w_bce"]),
            use_focal=bool(cfg["use_focal"]),
        ),
        metrics=[DiceBin(threshold=0.5, name="dice50"), IoUBin(threshold=0.5, name="iou50")],
    )
    return model

# ======================
# Metrics
# ======================
class DiceBin(Metric):
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

class IoUBin(Metric):
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

# ======================
# Confusion-matrix helpers
# ======================
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

def plot_confusion_matrix(tp, fp, fn, tn, out_path: Path, title="Confusion Matrix", class_names=None, normalize=False):
    if class_names is None:
        class_names = ("Background", "Leaf")
    
    # Build 2x2 matrix: [[TN, FP], [FN, TP]]
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    total = np.sum(cm)
    mat = cm.astype(np.float64)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        mat = mat / np.maximum(row_sums, 1e-12)
    
    # Create figure
    _, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(mat, interpolation='nearest', cmap='Blues', vmin=0)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Count' if normalize else 'Count', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Pred\n{class_names[0]}", f"Pred\n{class_names[1]}"])
    ax.set_yticklabels([f"True\n{class_names[0]}", f"True\n{class_names[1]}"])
    
    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), ha="center")
    
    # Add text annotations
    thresh = mat.max() / 2.0 if mat.size else 0.0
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            val = mat[i, j]
            text_color = "white" if val > thresh else "black"
            txt = f"{val:.2f}" if normalize else f"{count:,}\n({(count / max(total, 1) * 100.0):.1f}%)"
            ax.text(j, i, txt,
                   ha="center", va="center", color=text_color, fontsize=14, weight='bold')
    
    # Labels and title
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    
    # Add metrics text below the matrix
    metrics = cm_metrics(tp, fp, fn, tn)
    metrics_text = (
        f"Accuracy: {metrics['accuracy']:.4f}  |  "
        f"Precision: {metrics['precision']:.4f}  |  "
        f"Recall: {metrics['recall']:.4f}  |  "
        f"F1: {metrics['f1']:.4f}"
    )
    plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    ensure_dir(out_path.parent)
    plt.savefig(str(out_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] saved {'normalized ' if normalize else ''}confusion matrix -> {out_path}")

# ======================
# Overlay GRID
# ======================
LEAF_CLASSES_5 = [
    "ALTERNARIA LEAF SPOT",
    "LEAF SPOT (EARLY AND LATE)",
    "HEALTHY",
    "ROSETTE",
    "RUST",
]

_PATTERNS = [
    ("ALTERNARIA LEAF SPOT", re.compile(r"(?:^|_)ALTERNARIA_LEAF_SPOT(?:_|$)", re.IGNORECASE)),
    ("LEAF SPOT (EARLY AND LATE)", re.compile(r"(?:^|_)LEAF_SPOT_?\(EARLY_AND_LATE\)(?:_|$)", re.IGNORECASE)),
    ("HEALTHY", re.compile(r"(?:^|_)HEALTHY(?:_|$)", re.IGNORECASE)),
    ("ROSETTE", re.compile(r"(?:^|_)ROSETTE(?:_|$)", re.IGNORECASE)),
    ("RUST", re.compile(r"(?:^|_)RUST(?:_|$)", re.IGNORECASE)),
]

def infer_leaf_class_from_filename(path_str: str) -> str:
    stem = Path(path_str).stem
    for cname, pat in _PATTERNS:
        if pat.search(stem):
            return cname
    return "UNKNOWN"

def _to_uint8_rgb_from_bgr(bgr_uint8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB)

def _overlay_mask_rgb(rgb_uint8: np.ndarray, mask01: np.ndarray, color_rgb=(0, 255, 0), alpha=0.45) -> np.ndarray:
    img = rgb_uint8.astype(np.float32)
    m = (mask01 > 0.5).astype(np.float32)[..., None]
    col = np.array(color_rgb, dtype=np.float32)[None, None, :]
    out = img * (1.0 - m * alpha) + col * (m * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)

def _put_text(img_rgb: np.ndarray, text: str) -> np.ndarray:
    out = img_rgb.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 24), (0, 0, 0), thickness=-1)
    cv2.putText(out, text[:120], (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out

def _make_eval_tile(rgb_uint8: np.ndarray, gt01: np.ndarray, pr01: np.ndarray, title: str, alpha=0.45) -> np.ndarray:
    if gt01 is None:
        gt = _put_text(rgb_uint8.copy(), "GT: (missing)")
    else:
        gt = _overlay_mask_rgb(rgb_uint8, gt01, color_rgb=(0, 255, 0), alpha=alpha)
        gt = _put_text(gt, f"GT (green) | {title}")

    pr = _overlay_mask_rgb(rgb_uint8, pr01, color_rgb=(255, 0, 0), alpha=alpha)
    pr = _put_text(pr, f"PRED (red) | {title}")

    # Keep only GT and PRED panels for cleaner comparison.
    return np.concatenate([gt, pr], axis=1)

def save_overlay_grid_eval(val_pairs, model, out_path: Path, img_h: int, img_w: int,
    thr: float, n: int = 25, cols: int = 5, alpha: float = 0.45, seed: int = 42,):
    if not val_pairs:
        print("[WARN] val_pairs kosong. Skip overlay grid.")
        return

    rng = np.random.RandomState(int(seed))
    buckets = {c: [] for c in LEAF_CLASSES_5}
    unknown = []
    for ip, mp in val_pairs:
        c = infer_leaf_class_from_filename(ip)
        if c in buckets:
            buckets[c].append((ip, mp))
        else:
            unknown.append((ip, mp))

    for c in buckets:
        rng.shuffle(buckets[c])
    rng.shuffle(unknown)

    k = len(LEAF_CLASSES_5)
    n = int(n)
    cols = max(1, int(cols))
    if n <= 0:
        print("[WARN] grid_n <= 0. Skip overlay grid.")
        return

    base = n // k
    rem = n % k

    selected = []
    for i, cname in enumerate(LEAF_CLASSES_5):
        take = base + (1 if i < rem else 0)
        if take > 0:
            selected.extend(buckets[cname][:take])

    if len(selected) < n:
        leftovers = []
        for i, cname in enumerate(LEAF_CLASSES_5):
            used = base + (1 if i < rem else 0)
            leftovers.extend(buckets[cname][used:])
        leftovers.extend(unknown)
        rng.shuffle(leftovers)
        selected.extend(leftovers[: (n - len(selected))])

    selected = selected[:n]

    tiles = []
    for img_path, mask_path in selected:
        bgr0 = cv2.imread(img_path)
        if bgr0 is None:
            continue

        bgr = cv2.resize(bgr0, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        x = (bgr.astype(np.float32) / 255.0)[None, ...]

        prob = model.predict(x, verbose=0)[0, ..., 0]
        pr01 = (prob > float(thr)).astype(np.float32)

        gt01 = None
        gt0 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt0 is not None:
            gt = cv2.resize(gt0, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            gt01 = (gt > 0).astype(np.float32)

        rgb = _to_uint8_rgb_from_bgr(bgr)
        cls = infer_leaf_class_from_filename(img_path)
        title = f"[{cls}] {Path(img_path).name} | thr={float(thr):.2f}"
        tile = _make_eval_tile(rgb, gt01, pr01, title=title, alpha=float(alpha))
        tiles.append(tile)

    if not tiles:
        print("[WARN] Tidak ada tile. Skip overlay grid.")
        return

    # Slide-friendly output: 2x2 grid (4 tiles) per page.
    items_per_page = 4
    page_cols = 2
    page_rows = 2
    tile_h, tile_w = tiles[0].shape[:2]
    saved_pages = []
    out_stem = out_path.stem
    out_sfx = out_path.suffix if out_path.suffix else ".png"

    ensure_dir(out_path.parent)
    total_pages = int(np.ceil(len(tiles) / items_per_page))
    for page_idx in range(total_pages):
        start = page_idx * items_per_page
        end = min(start + items_per_page, len(tiles))
        page_tiles = list(tiles[start:end])

        pad_needed = items_per_page - len(page_tiles)
        if pad_needed > 0:
            blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            page_tiles.extend([blank] * pad_needed)

        rows_img = []
        for r in range(page_rows):
            row_tiles = page_tiles[r * page_cols:(r + 1) * page_cols]
            rows_img.append(np.concatenate(row_tiles, axis=1))
        grid = np.concatenate(rows_img, axis=0)

        page_path = out_path.parent / f"{out_stem}_page_{page_idx + 1:02d}{out_sfx}"
        cv2.imwrite(str(page_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        saved_pages.append(page_path)

    print(f"[OK] saved overlay grid pages -> {out_path.parent / (out_stem + '_page_XX' + out_sfx)} | pages={len(saved_pages)}")

    dist = {c: 0 for c in LEAF_CLASSES_5}
    dist["UNKNOWN"] = 0
    for ip, _ in selected:
        dist[infer_leaf_class_from_filename(ip)] = dist.get(infer_leaf_class_from_filename(ip), 0) + 1
    print("[GRID_DIST]", dist)
    return saved_pages

# ======================
# Tuning
# ======================
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
    return normalize_cfg(cfg)

def proxy_trial(train_pairs, val_pairs, args, cfg, trial_idx: int = 0):
    tf.keras.backend.clear_session()
    gc.collect()

    model = None
    tr_ds = None
    va_ds = None

    proxy_epochs = max(2, min(8, int(args.epochs)))

    try:
        model = build_unet(cfg, input_shape=(args.img_h, args.img_w, 3))
        model = compile_model(model, cfg)

        tr_ds = make_dataset(
            train_pairs,
            args.img_h,
            args.img_w,
            args.batch_size,
            shuffle=True,
            seed=int(args.seed) + int(trial_idx),
        )
        va_ds = make_dataset(
            val_pairs,
            args.img_h,
            args.img_w,
            args.batch_size,
            shuffle=False,
        )

        cb = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=max(1, int(args.patience) // 2),
                restore_best_weights=True,
                verbose=0,
            ),
        ]

        hist = model.fit(
            tr_ds,
            validation_data=va_ds,
            epochs=proxy_epochs,
            verbose=0,
            callbacks=cb,
        )
        return float(np.min(hist.history.get("val_loss", [np.inf])))

    except tf.errors.ResourceExhaustedError:
        return np.inf
    finally:
        cleanup(model, tr_ds, va_ds)

def tune(args):
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    train_pairs = build_pairs(Path(args.data_root) / "train" / "images", Path(args.data_root) / "train" / "masks")
    val_pairs   = build_pairs(Path(args.data_root) / "val"   / "images", Path(args.data_root) / "val"   / "masks")
    if not train_pairs or not val_pairs:
        raise SystemExit("train/val pairs not found. Check dataset path.")

    best_cfg = None
    best_val_loss = float("inf")

    log_path = out_dir / "severity_tuning_log.csv"
    best_cfg_path = out_dir / "best_severity_config.json"
    best_w_path = out_dir / "best_severity_model.weights.h5"

    fieldnames = [
        "trial", "val_loss",
        "base_filters", "depth", "use_bn", "dropout", "sep_conv",
        "lr", "optimizer", "w_dice", "w_bce", "use_focal"
    ]

    rng = random.Random(args.seed)

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for t in range(1, int(args.trials) + 1):
            cfg = sample_config(rng)

            vloss = proxy_trial(train_pairs, val_pairs, args, cfg, trial_idx=t)
            if np.isinf(vloss):
                print(f"[TUNE] trial {t} OOM -> skipped")
                continue

            row = {"trial": t, "val_loss": vloss, **cfg}
            w.writerow(row)
            f.flush()

            if vloss < best_val_loss:
                best_val_loss = vloss
                best_cfg = cfg

                model_best = compile_model(
                    build_unet(best_cfg, input_shape=(args.img_h, args.img_w, 3)),
                    best_cfg,
                )
                model_best.save_weights(str(best_w_path))
                cleanup(model_best)

                save_json(best_cfg, best_cfg_path)

            print(f"[TUNE] trial {t:03d}/{args.trials} | val_loss={vloss:.6f} | best={best_val_loss:.6f}")

    print("[DONE] tuning")
    print("[OK] best cfg :", best_cfg_path)
    print("[OK] best wts :", best_w_path)
    print("[OK] log      :", log_path)

# ======================
# Train
# ======================
def train(args):
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    cfg_path = Path(args.cfg_json) if args.cfg_json else (out_dir / "best_tuned_cfg.json")
    if not cfg_path.exists():
        raise SystemExit(
            "Config not found.\n"
            f"- Expected: {cfg_path}\n"
            "- Run tune first (creates best_tuned_cfg.json), or pass --cfg_json path.json"
        )
    cfg = normalize_cfg(load_json(cfg_path))

    train_pairs = build_pairs(Path(args.data_root) / "train" / "images", Path(args.data_root) / "train" / "masks")
    val_pairs   = build_pairs(Path(args.data_root) / "val"   / "images", Path(args.data_root) / "val"   / "masks")
    if not train_pairs or not val_pairs:
        raise SystemExit("train/val pairs not found. Check dataset path.")

    tf.keras.backend.clear_session()
    gc.collect()

    model = build_unet(cfg, input_shape=(args.img_h, args.img_w, 3))
    model = compile_model(model, cfg)

    tr_ds = make_dataset(train_pairs, args.img_h, args.img_w, args.batch_size, shuffle=True, seed=args.seed)
    va_ds = make_dataset(val_pairs,   args.img_h, args.img_w, args.batch_size, shuffle=False)

    best_w_path  = out_dir / "best_severity_model.weights.h5"

    hist_csv  = out_dir / "severity_train_history.csv"
    hist_json = out_dir / "severity_train_history.json"

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

    hist_clean = {k: [float(x) for x in v] for k, v in hist.history.items()}
    save_json(hist_clean, hist_json)
    print("[OK] saved history json:", hist_json)

    plot_training_curves(hist_clean, out_dir=out_dir, prefix="severity")

    cleanup(model, tr_ds, va_ds)

# ======================
# Eval
# ======================
def eval_model(args):
    out_dir = Path(RES_DIR)
    ensure_dir(out_dir)
    cfg_path = Path(args.cfg_json) if args.cfg_json else (Path(MODEL_DIR) / "best_tuned_cfg.json")
    weights_path = Path(args.weights) if args.weights else (Path(MODEL_DIR) / "best_severity_model.weights.h5")

    if not cfg_path.exists(): raise SystemExit(f"[ERROR] Config not found: {cfg_path}")
    if not weights_path.exists(): raise SystemExit(f"[ERROR] Weights not found: {weights_path}")

    val_pairs = build_pairs(Path(args.data_root) / "val" / "images", Path(args.data_root) / "val" / "masks")
    if not val_pairs: raise SystemExit("[ERROR] val pairs not found. Check dataset path.")

    tf.keras.backend.clear_session()
    gc.collect()
    cfg = normalize_cfg(load_json(cfg_path))
    model = compile_model(build_unet(cfg, input_shape=(args.img_h, args.img_w, 3)), cfg)
    model.load_weights(str(weights_path))
    print("[OK] loaded weights:", weights_path)
    print("[OK] loaded cfg    :", cfg_path)

    eval_dir = out_dir / "eval_outputs"
    pred_mask_dir = eval_dir / "pred_masks"
    masked_dir = eval_dir / "masked_pred"
    ensure_dir(eval_dir)
    ensure_dir(pred_mask_dir)
    ensure_dir(masked_dir)
    
    dice_m = DiceBin(threshold=args.threshold, name="dice50")
    iou_m  = IoUBin(threshold=args.threshold, name="iou50")
    loss_sum, loss_cnt = 0.0, 0
    agg_tp = agg_fp = agg_fn = agg_tn = 0
    per_image_rows = []

    for img_path, mask_path in tqdm(val_pairs, desc="EVAL", ncols=140):
        bgr0 = cv2.imread(img_path)
        if bgr0 is None:
            continue

        bgr = cv2.resize(bgr0, (args.img_w, args.img_h), interpolation=cv2.INTER_LINEAR)
        x = (bgr.astype(np.float32) / 255.0)[None, ...]
        prob = model.predict(x, verbose=0)[0, ..., 0]
        pred255 = (prob > args.threshold).astype(np.uint8) * 255
        stem = Path(img_path).stem
        cv2.imwrite(str(pred_mask_dir / f"{stem}.png"), pred255)
        m01 = (pred255 > 0).astype(np.uint8)
        pred_back = cv2.resize(m01, (bgr0.shape[1], bgr0.shape[0]), interpolation=cv2.INTER_NEAREST)
        masked = bgr0 * pred_back[:, :, None]
        cv2.imwrite(str(masked_dir / f"{stem}.jpg"), masked)
        gt0 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        dice_i = iou_i = ""
        tp = fp = fn = tn = ""

        if gt0 is not None:
            gt = cv2.resize(gt0, (args.img_w, args.img_h), interpolation=cv2.INTER_NEAREST)
            gt_bin = (gt > 0).astype(np.float32)
            pr_bin = (prob > args.threshold).astype(np.float32)
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
            dice_m.update_state(gt_bin[None, ..., None], pr_bin[None, ..., None])
            iou_m.update_state(gt_bin[None, ..., None], pr_bin[None, ..., None])
            inter = float((gt_bin * pr_bin).sum())
            denom = float(gt_bin.sum() + pr_bin.sum())
            dice_i = (2.0 * inter + 1.0) / (denom + 1.0 + 1e-9)
            union = float(((gt_bin + pr_bin) > 0).sum())
            iou_i = (inter + 1.0) / (union + 1.0 + 1e-9)

            if args.cm_enable:
                gt01 = (gt_bin > 0.5).astype(np.uint8)
                pr01 = (pr_bin > 0.5).astype(np.uint8)
                tp, fp, fn, tn = cm_from_binary(gt01, pr01)
                agg_tp += tp
                agg_fp += fp
                agg_fn += fn
                agg_tn += tn
        row = {
            "filename": Path(img_path).name, "dice": float(dice_i) if dice_i != "" else "", "iou":  float(iou_i)  if iou_i  != "" else "",
        }
        if args.cm_enable:
            row.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
        per_image_rows.append(row)

    dice = float(dice_m.result().numpy())
    iou  = float(iou_m.result().numpy())
    avg_loss = float(loss_sum / max(loss_cnt, 1))
    weights_stem = weights_path.stem
    thr_tag = f"{float(args.threshold):.2f}".replace(".", "p")
    summary = {
        "weights": str(weights_path), "cfg_path": str(cfg_path), "threshold": float(args.threshold), "loss": float(avg_loss),
        "dice50": float(dice), "iou50": float(iou), "n_val": int(len(val_pairs)),
    }

    if args.cm_enable:
        cm_sum = cm_metrics(agg_tp, agg_fp, agg_fn, agg_tn)
        summary.update({
            "cm_tp": cm_sum["tp"], "cm_fp": cm_sum["fp"], "cm_fn": cm_sum["fn"], "cm_tn": cm_sum["tn"],
            "cm_precision": cm_sum["precision"], "cm_recall": cm_sum["recall"], "cm_f1": cm_sum["f1"],
            "cm_accuracy": cm_sum["accuracy"], "cm_specificity": cm_sum["specificity"],
        })
        
        # Save confusion matrix visualization
        cm_png = eval_dir / f"eval_confusion_matrix_{weights_stem}_thr{thr_tag}.png"
        plot_confusion_matrix(
            tp=agg_tp, fp=agg_fp, fn=agg_fn, tn=agg_tn,
            out_path=cm_png,
            title=f"Confusion Matrix (threshold={args.threshold:.2f})",
            class_names=("Background", "Leaf Mask"),
            normalize=False
        )

        cm_norm_png = eval_dir / f"eval_confusion_matrix_normalized_{weights_stem}_thr{thr_tag}.png"
        plot_confusion_matrix(
            tp=agg_tp, fp=agg_fp, fn=agg_fn, tn=agg_tn,
            out_path=cm_norm_png,
            title=f"Confusion Matrix - Normalized (threshold={args.threshold:.2f})",
            class_names=("Background", "Leaf Mask"),
            normalize=True
        )


    if args.save_grid_overlay:
        grid_path = eval_dir / f"eval_overlay_grid_{weights_stem}_thr{thr_tag}.png"
        save_overlay_grid_eval(
            val_pairs=val_pairs, model=model, out_path=grid_path, img_h=int(args.img_h), img_w=int(args.img_w),
            thr=float(args.threshold), n=int(args.grid_n), cols=int(args.grid_cols), alpha=float(args.grid_alpha),
            seed=int(args.seed),
        )
    print("[DONE] eval")
    cleanup(model)

# ======================
# CLI
# ======================
def parse_args():
    p = argparse.ArgumentParser("Severity leaf mask segmentation (binary UNet)")

    p.add_argument("--mode", choices=["tune", "train", "eval"], required=True)

    p.add_argument("--data_root", type=str, default=BASE_DIR)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--cfg_json", type=str, default="", help="Optional cfg json (train/eval). If empty, uses default locations.")
    p.add_argument("--img_h", type=int, default=480)
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    p.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision (if supported).")

    p.add_argument("--trials", type=int, default=30)

    p.add_argument("--weights", type=str, default="")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--cm_enable", action="store_true")

    p.add_argument("--save_grid_overlay", action="store_true",
                   help="Jika diaktifkan, eval akan menyimpan 1 gambar overlay grid untuk dokumentasi.")
    p.add_argument("--grid_n", type=int, default=25,
                   help="Jumlah sample untuk grid (default 25 = 5 kelas x 5).")
    p.add_argument("--grid_cols", type=int, default=5,
                   help="Jumlah kolom grid (default 5).")
    p.add_argument("--grid_alpha", type=float, default=0.45,
                   help="Alpha overlay mask (0..1).")

    return p.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)
    set_memory_growth()
    enable_mixed_precision(bool(args.mixed_precision))

    if args.mode in ["tune", "train"]:
        ensure_dir(Path(args.out_dir))

    if args.mode == "tune":
        tune(args)
    elif args.mode == "train":
        train(args)
    else:
        eval_model(args)

if __name__ == "__main__":
    main()