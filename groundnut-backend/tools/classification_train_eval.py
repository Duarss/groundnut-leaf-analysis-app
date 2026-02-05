# exe_classification_model.py
#
# ============================================================
# EfficientNetB4 Image Classification (TA)
# Joint Search (HEAD ARCH + HEAD HP + LR1 + LR2 + UNFREEZE)
# Search methods: RANDOM or GRID
# ============================================================
#
# Inti desain (sidang-proof, ringkas):
# - Backbone: EfficientNetB4 pretrained ImageNet (include_top=False) => feature extractor.
# - Head: pooling -> dropout -> dense -> dropout -> softmax (opsional dense2) => classifier 5 kelas.
# - Training 2 fase:
#   (1) Phase1: backbone frozen, latih head saja (stabil, hindari rusak fitur pretrained).
#   (2) Phase2: fine-tune sebagian layer terakhir backbone + BN dibekukan (stabil untuk batch kecil).
# - Pemilihan config murni dari tuning (random/grid) pada domain diskret; tidak ada baseline/anchor fallback.
# - Best config dipilih berdasarkan proxy best_val_loss (tie-break: best_val_acc).
#
# Catatan operasional:
# - Grid search bersifat exhaustive pada domain => --trials DIABAIKAN saat grid untuk menghindari bias subset.
# - Untuk grid yang terlalu besar, kecilkan domain list lewat CLI (mis. 1-2 nilai per dimensi).
# - eval/predict wajib punya best_tuned_cfg.json (hasilkan dulu via train/search).

import os, json, argparse, random
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, GlobalMaxPooling2D,
    Dropout, BatchNormalization, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K


# ========= Performance tweaks =========
# Mixed precision mempercepat training & hemat VRAM (khusus GPU yang mendukung).
# Policy global mixed_float16, namun output head dipaksa float32 untuk stabilitas softmax/loss.
mixed_precision.set_global_policy('mixed_float16')

# Memory growth mencegah TF "mengambil" seluruh VRAM di awal.
gpus = tf.config.experimental.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass


# ========= Reproducibility =========
# Seed konsisten untuk random search & determinisme relatif pada pipeline.
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)


# ========= Paths =========
BASE_DIR  = "utils/classification_dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train_balanced")
VAL_DIR   = os.path.join(BASE_DIR, "val_balanced")
MODEL_DIR = "models/classification"
os.makedirs(MODEL_DIR, exist_ok=True)


# ========= Fixed params =========
# Batch size disatukan untuk train/eval agar pipeline konsisten; turunkan jika OOM.
BATCH_SIZE = 16

# EfficientNetB4 default input size 380x380 agar selaras dengan pretraining.
IMG_SIZE = (380, 380)

NUM_CLASSES = 5

# Jadwal training final (tetap, tuning hanya mengubah lr & unfreeze).
EPOCHS_PHASE1 = 3
EPOCHS_PHASE2 = 45

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic")

# ========= Data =========
def make_train_datagen():
    # Konservatif: hanya preprocess_input (tanpa augment besar) agar stabil & reproducible.
    return ImageDataGenerator(preprocessing_function=preprocess_input)

def make_eval_datagen():
    # Evaluasi deterministik.
    return ImageDataGenerator(preprocessing_function=preprocess_input)


# ========= Model builders (HEAD ARCH TUNING) =========
def _apply_head(x, cfg_head, num_classes, activation, dtype_last):
    """
    Head classifier di atas backbone (feature maps HxWxC -> logits/prob 5 kelas).

    Keluarga head (dituning):
    - baseline: GAP -> Dropout -> Dense(ReLU) -> Dropout -> Softmax
    - swish   : sama, tapi Dense(Swish) (kadang gradien lebih halus)
    - mlp2    : baseline + Dense kedua (kapasitas lebih tinggi, risiko overfit naik)
    - gap_gmp : concat(GAP, GMP) -> Dropout -> Dense -> Dropout -> Softmax
               (GMP menangkap aktivasi puncak; GAP menangkap rata-rata global)
    """
    head_type = str(cfg_head["head_type"])

    # Hyperparameter head dituning via random/grid.
    drop1 = float(cfg_head["drop1"])
    drop2 = float(cfg_head["drop2"])
    dense_units = int(cfg_head["dense_units"])
    dense_l2 = float(cfg_head.get("dense_l2", 0.0))

    # L2 untuk menahan bobot Dense agar tidak overfit.
    reg = l2(dense_l2) if dense_l2 > 0 else None

    # Pooling: merangkum feature maps jadi vektor fitur global (lebih robust daripada Flatten).
    if head_type == "gap_gmp":
        gap = GlobalAveragePooling2D(name="gap")(x)  # ringkas rata-rata tiap channel
        gmp = GlobalMaxPooling2D(name="gmp")(x)      # ringkas puncak aktivasi tiap channel
        x = Concatenate(name="gap_gmp_concat")([gap, gmp])  # gabungkan info global+peak
    else:
        x = GlobalAveragePooling2D(name="gap")(x)

    # Aktivasi Dense: baseline pakai relu; swish/mlp2 pakai swish.
    act = "relu"
    if head_type in ("swish", "mlp2"):
        act = "swish"

    # Dropout sebelum Dense: regularisasi fitur global.
    x = Dropout(drop1, name="drop1")(x)

    # Dense classifier: memetakan fitur backbone ke representasi yang cocok untuk dataset.
    x = Dense(
        dense_units,
        activation=act,
        kernel_regularizer=reg,
        name="dense"
    )(x)

    # Dropout setelah Dense: mencegah head terlalu percaya pada neuron tertentu.
    x = Dropout(drop2, name="drop2")(x)

    # Opsional Dense kedua (mlp2): tambah kapasitas, biasanya perlu dropout tambahan.
    if head_type == "mlp2":
        dense_units2 = int(cfg_head.get("dense_units2", max(64, dense_units // 2)))
        drop3 = float(cfg_head.get("drop3", 0.2))
        x = Dense(
            dense_units2,
            activation=act,
            kernel_regularizer=reg,
            name="dense2"
        )(x)
        x = Dropout(drop3, name="drop3")(x)

    # Output: Softmax 5 kelas, dtype float32 untuk stabilitas numerik pada mixed precision.
    out = Dense(num_classes, activation=activation, dtype=dtype_last, name="predictions")(x)
    return out

def build_model_from_cfg(cfg, num_classes=NUM_CLASSES, img_size=IMG_SIZE, activation="softmax", dtype_last="float32"):
    """
    Backbone pretrained:
    - EfficientNetB4 weights=imagenet, include_top=False => hanya feature extractor.
    - Top/classifier ImageNet (1000 kelas) dibuang, diganti head custom (5 kelas).
    """
    base = EfficientNetB4(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size[0], img_size[1], 3)
    )
    out = _apply_head(base.output, cfg["head"], num_classes, activation=activation, dtype_last=dtype_last)
    model_ = Model(inputs=base.input, outputs=out, name="EffNetB4_Classifier")
    return model_, base

def freeze_batchnorm_layers(model):
    # BatchNorm sensitif terhadap batch kecil; saat fine-tuning sering dibekukan agar stabil.
    for layer in model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False


# ========= Utilities =========
def load_class_indices(path):
    with open(path, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return class_indices, idx_to_class

def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_single_image(path, img_size=IMG_SIZE):
    # Load 1 gambar untuk prediksi single image.
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    if not path.lower().endswith(VALID_EXTS):
        raise ValueError(f"Unsupported image extension for: {path}")
    img = tf.keras.utils.load_img(path, target_size=img_size)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr.astype(np.float32))
    return arr

def topk_from_probs(probs_row, idx_to_class, k=5):
    # Ambil Top-K label dari probabilitas softmax.
    k = min(k, probs_row.shape[-1])
    idx = probs_row.argsort()[-k:][::-1]
    return [(idx_to_class[i], float(probs_row[i])) for i in idx]



# ========= Metrics & artifacts =========
def confusion_and_report(true_idx, pred_idx, num_classes, idx_to_class=None):
    # Hitung confusion matrix + per-class metrics (precision/recall/f1) tanpa sklearn.
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_idx, pred_idx):
        cm[int(t), int(p)] += 1

    eps = 1e-12
    support = cm.sum(axis=1)
    tp = np.diag(cm).astype(np.float64)
    precision = tp / np.maximum(cm.sum(axis=0), eps)
    recall    = tp / np.maximum(cm.sum(axis=1), eps)
    f1        = 2 * precision * recall / np.maximum(precision + recall, eps)

    macro_p = float(np.mean(precision))
    macro_r = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))
    acc = float(tp.sum() / np.maximum(cm.sum(), 1))

    names = [idx_to_class[i] if idx_to_class else str(i) for i in range(num_classes)]
    print("\nPer-class metrics:")
    for i, name in enumerate(names):
        print(f"  {name:>28s} | P:{precision[i]:.3f}  R:{recall[i]:.3f}  F1:{f1[i]:.3f}  Support:{int(support[i])}")
    print(f"\nOverall: Acc:{acc:.4f}  MacroP:{macro_p:.4f}  MacroR:{macro_r:.4f}  MacroF1:{macro_f1:.4f}")

    return cm, {
        "precision": precision, "recall": recall, "f1": f1,
        "macro_p": macro_p, "macro_r": macro_r, "macro_f1": macro_f1, "acc": acc, "support": support
    }

def save_confusion_csv(cm, idx_to_class, path):
    # Simpan confusion matrix ke CSV.
    labels = [idx_to_class[i] for i in range(cm.shape[0])] if idx_to_class else [str(i) for i in range(cm.shape[0])]
    df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=True)

def save_report_csv(report_dict, idx_to_class, path):
    # Simpan per-class precision/recall/f1/support + overall macro ke CSV.
    rows = []
    for i in range(len(report_dict["precision"])):
        name = idx_to_class[i] if idx_to_class else str(i)
        rows.append({
            "class": name,
            "precision": float(report_dict["precision"][i]),
            "recall": float(report_dict["recall"][i]),
            "f1": float(report_dict["f1"][i]),
            "support": int(report_dict["support"][i]),
        })
    rows.append({
        "class": "OVERALL",
        "precision": float(report_dict["macro_p"]),
        "recall": float(report_dict["macro_r"]),
        "f1": float(report_dict["macro_f1"]),
        "support": int(np.sum(report_dict["support"])),
    })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)

def plot_confusion_png(cm, idx_to_class, out_png, normalize=False):
    # Plot confusion matrix PNG (raw dan normalized).
    labels = [idx_to_class[i] for i in range(cm.shape[0])] if idx_to_class else [str(i) for i in range(cm.shape[0])]
    mat = cm.astype(np.float64)
    if normalize:
        mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1e-12)

    plt.figure(figsize=(9, 7))
    plt.imshow(mat)
    plt.title("Confusion Matrix (Normalized)" if normalize else "Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = f"{v:.2f}" if normalize else str(int(v))
            plt.text(j, i, txt, ha="center", va="center", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

def merge_histories(histories):
    """
    Gabungkan beberapa Keras History (mis. phase-1 dan phase-2) jadi satu dict list.
    """
    merged = {}
    for h in histories:
        if h is None:
            continue
        d = getattr(h, "history", None)
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            merged.setdefault(k, [])
            merged[k].extend(list(v))
    return merged

def _to_python_float(x):
    """
    Convert numpy / tf scalar to native Python float (JSON-safe).
    """
    try:
        if isinstance(x, (np.generic,)):
            return x.item()
        if tf.is_tensor(x):
            return float(x.numpy())
        return float(x)
    except Exception:
        return x

def save_history(history_dict, out_dir, prefix="classification"):
    os.makedirs(out_dir, exist_ok=True)

    # === convert to JSON-safe types ===
    safe_hist = {}
    for k, v in history_dict.items():
        if isinstance(v, (list, tuple)):
            safe_hist[k] = [_to_python_float(x) for x in v]
        else:
            safe_hist[k] = _to_python_float(v)

    # JSON
    json_path = os.path.join(out_dir, f"{prefix}_train_history.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(safe_hist, f, indent=2)
    print(f"[INFO] Saved training history JSON: {json_path}")

    # CSV (pandas aman dengan numpy float)
    csv_path = os.path.join(out_dir, f"{prefix}_train_history.csv")
    pd.DataFrame(safe_hist).to_csv(csv_path, index=False)
    print(f"[INFO] Saved training history CSV: {csv_path}")


def plot_training_curves(history_dict, out_dir, prefix="classification"):
    """
    Buat 2 grafik:
    1) loss: train loss vs val loss
    2) accuracy: train acc vs val acc
    """
    os.makedirs(out_dir, exist_ok=True)

    if not history_dict:
        print("[WARN] Empty history dict. Skip plotting.")
        return

    # cari panjang epoch dari key pertama yang non-empty
    n_epochs = None
    for k, v in history_dict.items():
        if isinstance(v, (list, tuple)) and len(v) > 0:
            n_epochs = len(v)
            break
    if not n_epochs:
        print("[WARN] History has no epoch data. Skip plotting.")
        return

    epochs = np.arange(1, n_epochs + 1)

    # --- LOSS CURVE ---
    if "loss" in history_dict:
        plt.figure()
        plt.plot(epochs, history_dict.get("loss", []), label="train_loss")
        if "val_loss" in history_dict:
            plt.plot(epochs, history_dict.get("val_loss", []), label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{prefix}_curve_loss.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[INFO] Saved plot: {out_path}")
    else:
        print("[WARN] 'loss' not found in history. Skip loss plot.")

    # --- ACCURACY CURVE ---
    acc_key = "accuracy" if "accuracy" in history_dict else ("acc" if "acc" in history_dict else None)
    if acc_key is not None:
        val_acc_key = "val_accuracy" if "val_accuracy" in history_dict else ("val_acc" if "val_acc" in history_dict else None)

        plt.figure()
        plt.plot(epochs, history_dict.get(acc_key, []), label="train_accuracy")
        if val_acc_key is not None:
            plt.plot(epochs, history_dict.get(val_acc_key, []), label="val_accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{prefix}_curve_accuracy.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[INFO] Saved plot: {out_path}")
    else:
        print("[WARN] accuracy metric not found in history (accuracy/acc). Skip accuracy plot.")

def save_true_vs_pred_grid(filepaths, y_true, y_pred, probs, idx_to_class, out_png,
                           max_items=16, samples_per_class=None, seed=GLOBAL_SEED):
    # Visualisasi stratified True-vs-Pred grid (ambil beberapa sampel tiap kelas; salah diprioritaskan).
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    probs = np.asarray(probs)

    n = min(len(y_true), len(y_pred), probs.shape[0], len(filepaths))
    if n <= 0:
        print("⚠️ No samples to plot for True vs Pred grid.")
        return

    y_true = y_true[:n]
    y_pred = y_pred[:n]
    probs = probs[:n]
    filepaths = list(filepaths[:n])

    conf = np.max(probs, axis=1)
    num_classes = int(probs.shape[1])

    if samples_per_class is None:
        samples_per_class = max(1, int(max_items) // max(1, num_classes))
    else:
        samples_per_class = max(1, int(samples_per_class))

    chosen = []
    chosen_set = set()

    for c in range(num_classes):
        idx_c = np.where(y_true == c)[0]
        if idx_c.size == 0:
            continue

        wrong_c = idx_c[y_pred[idx_c] != c]
        right_c = idx_c[y_pred[idx_c] == c]

        rng = np.random.default_rng(int(seed) + 1000 + c)
        rng.shuffle(wrong_c)
        rng.shuffle(right_c)

        pick = np.concatenate([wrong_c, right_c])[:samples_per_class]
        for i in pick.tolist():
            if i not in chosen_set:
                chosen.append(int(i))
                chosen_set.add(int(i))

    if len(chosen) < int(max_items):
        wrong_all = np.where(y_true != y_pred)[0]
        right_all = np.where(y_true == y_pred)[0]
        rng = np.random.default_rng(int(seed) + 9999)
        rng.shuffle(wrong_all)
        rng.shuffle(right_all)
        fill = np.concatenate([wrong_all, right_all]).tolist()
        for i in fill:
            if len(chosen) >= int(max_items):
                break
            if int(i) not in chosen_set:
                chosen.append(int(i))
                chosen_set.add(int(i))

    if len(chosen) == 0:
        print("⚠️ No samples to plot for True vs Pred grid.")
        return

    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    plt.figure(figsize=(12, 3 * rows))

    for k, idx in enumerate(chosen, start=1):
        img = tf.keras.utils.load_img(filepaths[idx], target_size=IMG_SIZE)
        plt.subplot(rows, cols, k)
        plt.imshow(img)
        tname = idx_to_class[int(y_true[idx])] if idx_to_class else str(int(y_true[idx]))
        pname = idx_to_class[int(y_pred[idx])] if idx_to_class else str(int(y_pred[idx]))
        plt.title(f"T:{tname}\nP:{pname} ({conf[idx]:.2f})", fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


# ========= Joint Search Utilities =========
def cfg_key(cfg):
    # Key unik untuk dedup kandidat (mencegah trial identik).
    h = cfg["head"]; s = cfg["schedule"]
    return (
        str(h["head_type"]),
        int(h["dense_units"]), float(h["drop1"]), float(h["drop2"]), float(h.get("dense_l2", 0.0)),
        int(h.get("dense_units2", 0)), float(h.get("drop3", 0.0)),
        float(s["lr_phase1"]), float(s["lr_phase2"]), int(s["unfreeze_last_n_layers"])
    )

def sample_one_cfg(dom, rng):
    # Random search: sampling satu konfigurasi dari domain diskret.
    head_type = str(rng.choice(dom["head_type"]))
    cfg = {
        "head": {
            "head_type": head_type,
            "dense_units": int(rng.choice(dom["dense_units"])),
            "drop1": float(rng.choice(dom["drop1"])),
            "drop2": float(rng.choice(dom["drop2"])),
            "dense_l2": float(rng.choice(dom["dense_l2"])),
            "dense_units2": int(rng.choice(dom["dense_units2"])),
            "drop3": float(rng.choice(dom["drop3"])),
        },
        "schedule": {
            "lr_phase1": float(rng.choice(dom["lr_phase1"])),
            "lr_phase2": float(rng.choice(dom["lr_phase2"])),
            "unfreeze_last_n_layers": int(rng.choice(dom["unfreeze_last_n_layers"])),
        }
    }
    return cfg

def build_search_domain_from_args(args):
    # Domain search ditentukan lewat CLI (list nilai diskret).
    return {
        "head_type": args.head_type,
        "dense_units": args.dense_units,
        "drop1": args.drop1,
        "drop2": args.drop2,
        "dense_l2": args.dense_l2,
        "dense_units2": args.dense_units2,
        "drop3": args.drop3,
        "lr_phase1": args.lr1,
        "lr_phase2": args.lr2,
        "unfreeze_last_n_layers": args.unfreeze,
    }

def generate_candidates_random(dom, trials, rng):
    # Random search: menghasilkan sejumlah kandidat acak.
    return [sample_one_cfg(dom, rng) for _ in range(int(trials))]

def generate_candidates_grid(dom):
    # Grid search: exhaustive semua kombinasi domain.
    keys = [
        "head_type", "dense_units", "drop1", "drop2", "dense_l2",
        "dense_units2", "drop3",
        "lr_phase1", "lr_phase2", "unfreeze_last_n_layers"
    ]
    values = [dom[k] for k in keys]

    cands = []
    for (
        head_type, dense_units, drop1, drop2, dense_l2,
        dense_units2, drop3,
        lr_phase1, lr_phase2, unfreeze_last
    ) in product(*values):
        cfg = {
            "head": {
                "head_type": str(head_type),
                "dense_units": int(dense_units),
                "drop1": float(drop1),
                "drop2": float(drop2),
                "dense_l2": float(dense_l2),
                "dense_units2": int(dense_units2),
                "drop3": float(drop3),
            },
            "schedule": {
                "lr_phase1": float(lr_phase1),
                "lr_phase2": float(lr_phase2),
                "unfreeze_last_n_layers": int(unfreeze_last),
            }
        }
        cands.append(cfg)
    return cands


def proxy_train_eval(train_gen, val_gen, cfg, proxy_phase1_epochs, proxy_phase2_epochs, proxy_patience):
    """
    Proxy training: training singkat untuk ranking kandidat (bukan hasil final).
    Metric seleksi: best val_loss (tie-break val_acc) dari training proxy.
    """
    K.clear_session()

    model, base_model = build_model_from_cfg(cfg, num_classes=train_gen.num_classes, img_size=IMG_SIZE)
    steps_per_epoch = int(np.ceil(train_gen.samples / BATCH_SIZE))
    val_steps = int(np.ceil(val_gen.samples / val_gen.batch_size))
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # Phase1 proxy: latih head saja (backbone frozen)
    base_model.trainable = False
    opt1 = Adam(learning_rate=float(cfg["schedule"]["lr_phase1"]), clipnorm=1.0)
    model.compile(optimizer=opt1, loss=loss_fn, metrics=["accuracy"])
    model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        epochs=int(proxy_phase1_epochs),
        verbose=0,
        workers=1,
        use_multiprocessing=False
    )

    # Phase2 proxy: fine-tune sebagian layer terakhir backbone, BN dibekukan agar stabil.
    base_model.trainable = True
    unfreeze_last = int(cfg["schedule"]["unfreeze_last_n_layers"])
    freeze_until = max(0, len(base_model.layers) - unfreeze_last)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True
    freeze_batchnorm_layers(base_model)

    opt2 = Adam(learning_rate=float(cfg["schedule"]["lr_phase2"]), clipnorm=1.0)
    model.compile(optimizer=opt2, loss=loss_fn, metrics=["accuracy"])

    # EarlyStopping pada proxy untuk hemat waktu dan ambil bobot terbaik proxy.
    es = EarlyStopping(
        monitor="val_loss",
        patience=int(proxy_patience),
        min_delta=1e-5,
        restore_best_weights=True,
        verbose=0
    )

    hist = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        epochs=int(proxy_phase2_epochs),
        callbacks=[es],
        verbose=0,
        workers=1,
        use_multiprocessing=False
    )

    h = hist.history
    best_val_loss = float(np.min(h.get("val_loss", [np.inf])))
    best_val_acc  = float(np.max(h.get("val_accuracy", [0.0])))

    K.clear_session()
    return best_val_loss, best_val_acc


def tune_cfg_cand_search(train_gen, val_gen, dom, trials, proxy_phase1_epochs, proxy_phase2_epochs, proxy_patience, method="random"):
    """
    Joint search (random/grid):
    - random: sampling sebanyak --trials
    - grid  : exhaustive (abaikan --trials agar tidak bias subset)
    Output: best_cfg + trials dataframe (CSV).
    """
    rng = np.random.default_rng(GLOBAL_SEED)

    if method == "grid":
        cands = generate_candidates_grid(dom)  # exhaustive
    elif method == "random":
        cands = generate_candidates_random(dom, trials, rng)
    else:
        raise ValueError(f"Unknown search method: {method}")

    # Dedup kandidat (misal domain list punya duplikasi).
    uniq = {}
    for c in cands:
        uniq[cfg_key(c)] = c
    cands = list(uniq.values())

    if len(cands) == 0:
        raise RuntimeError("No candidates generated. Check domain lists / trials.")

    # Warning runtime: grid bisa besar jika domain list besar.
    if method == "grid" and len(cands) > 2000:
        print(f"⚠️  Grid size is large: {len(cands)} candidates. "
              f"Reduce domain list sizes via CLI to keep runtime reasonable.")

    print(f"\n🎲 Tune candidates ({method}): {len(cands)}")

    rows = []
    for i, cfg in enumerate(cands, start=1):
        print(f"  Trial {i}/{len(cands)} | head_type={cfg['head']['head_type']}")

        vloss, vacc = proxy_train_eval(
            train_gen, val_gen, cfg,
            proxy_phase1_epochs=proxy_phase1_epochs,
            proxy_phase2_epochs=proxy_phase2_epochs,
            proxy_patience=proxy_patience
        )
        rows.append({
            "head_type": cfg["head"]["head_type"],
            "dense_units": cfg["head"]["dense_units"],
            "drop1": cfg["head"]["drop1"],
            "drop2": cfg["head"]["drop2"],
            "dense_l2": cfg["head"]["dense_l2"],
            "dense_units2": cfg["head"].get("dense_units2", 0),
            "drop3": cfg["head"].get("drop3", 0.0),
            "lr_phase1": cfg["schedule"]["lr_phase1"],
            "lr_phase2": cfg["schedule"]["lr_phase2"],
            "unfreeze_last_n_layers": cfg["schedule"]["unfreeze_last_n_layers"],
            "best_val_loss": vloss,
            "best_val_acc": vacc,
        })

    # Ranking: val_loss terendah (utama), val_acc tertinggi (tie-break).
    df = pd.DataFrame(rows).sort_values(["best_val_loss", "best_val_acc"], ascending=[True, False])

    # Simpan semua trial untuk transparansi paper/sidang.
    trials_csv = os.path.join(MODEL_DIR, "tune_cfg_cand_trials.csv")
    df.to_csv(trials_csv, index=False)
    print(f"📝 Saved joint trials -> {trials_csv}")

    # Ambil best candidate (ranking teratas).
    best = df.iloc[0].to_dict()
    best_cfg = {
        "head": {
            "head_type": str(best["head_type"]),
            "dense_units": int(best["dense_units"]),
            "drop1": float(best["drop1"]),
            "drop2": float(best["drop2"]),
            "dense_l2": float(best["dense_l2"]),
            "dense_units2": int(best.get("dense_units2", 128)),
            "drop3": float(best.get("drop3", 0.2)),
        },
        "schedule": {
            "lr_phase1": float(best["lr_phase1"]),
            "lr_phase2": float(best["lr_phase2"]),
            "unfreeze_last_n_layers": int(best["unfreeze_last_n_layers"]),
        }
    }
    return best_cfg, df, trials_csv


# ========= Spec (metadata untuk paper) =========
def save_spec(best_weights_path, last_weights_path, class_indices_path, best_tuned_cfg, search_meta):
    # Spec untuk dokumentasi: arsitektur, config terpilih, domain search, dan setting training.
    scientific_notes = {
        "selection_rationale": (
            "Konfigurasi dipilih murni dari hasil pencarian (random/grid) pada domain yang didefinisikan. "
            "Kandidat terbaik ditentukan berdasarkan proxy val_loss terendah (tie-break: val_acc)."
        ),
        "why_random_search": (
            "Random search efisien pada ruang kombinasi berdimensi banyak. Dengan budget trial terbatas, "
            "random search sering menemukan konfigurasi baik lebih cepat daripada grid search."
        ),
        "why_grid_search": (
            "Grid search mengevaluasi semua kombinasi pada domain diskret yang ditentukan, berguna ketika "
            "domain sudah dipersempit dan ingin exhaustive comparison."
        ),
        "why_tune_lr_and_unfreeze": (
            "LR fase 1 untuk optimisasi head saat backbone frozen, LR fase 2 untuk fine-tuning backbone. "
            "Jumlah layer yang di-unfreeze mengontrol trade-off adaptasi domain vs overfitting."
        ),
        "batch_size_rationale": (
            "Batch size diset tetap untuk konsistensi pipeline dan kompatibilitas VRAM 4GB."
        ),
        "proxy_training_rationale": (
            "Proxy training digunakan untuk ranking kandidat secara murah; bukan untuk performa final."
        ),
        "why_head_family_is_limited": (
            "Domain head dibatasi ke variasi ringan agar risiko degradasi rendah dan mudah dijelaskan."
        )
    }

    spec = {
        "backbone": "tf.keras.applications.EfficientNetB4",
        "img_size": list(IMG_SIZE),
        "batch_size": int(BATCH_SIZE),
        "num_classes": int(NUM_CLASSES),
        "best_tuned_cfg": best_tuned_cfg,
        "search_meta": search_meta,
        "train_schedule": {
            "epochs_phase1": int(EPOCHS_PHASE1),
            "epochs_phase2": int(EPOCHS_PHASE2),
        },
        "scientific_notes": scientific_notes,
        "paths": {
            "best_weights": best_weights_path,
            "final_weights": last_weights_path,
            "class_indices": class_indices_path,
        }
    }
    with open(os.path.join(MODEL_DIR, "classification_model_spec.json"), "w") as f:
        json.dump(spec, f, indent=2)
    print("✅ Wrote models/classification_model_spec.json")


# ========= CLI =========


# ========= Train =========
def train_mode(args):
    K.clear_session()

    # Data generator (preprocess_input) agar selaras dengan preprocessing EfficientNet.
    train_datagen = make_train_datagen()
    eval_datagen  = make_eval_datagen()

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=True, seed=GLOBAL_SEED
    )

    val_gen = eval_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False, seed=GLOBAL_SEED
    )

    # Simpan mapping label -> index untuk eval/predict.
    class_indices_path = os.path.join(MODEL_DIR, "classification_class_indices.json")
    with open(class_indices_path, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    best_tuned_path = os.path.join(MODEL_DIR, "best_tuned_cfg.json")
    search_meta = {"method": args.search_method, "cache_used": False}

    # cache-or-search:
    # - Jika ada best_tuned_cfg.json dan tidak force_search => reuse cfg (reproducible).
    # - Jika tidak ada / force_search => lakukan tuning (pure).
    if os.path.exists(best_tuned_path) and (not args.force_search):
        with open(best_tuned_path, "r") as f:
            best_tuned_cfg = json.load(f)
        search_meta["cache_used"] = True
        search_meta["selection_rule"] = "use cached best_tuned_cfg.json"
        print(f"✅ Using cached best tuned cfg -> {best_tuned_path}")
    else:
        dom = build_search_domain_from_args(args)

        best_cfg, df, trials_csv = tune_cfg_cand_search(
            train_gen, val_gen,
            dom=dom,
            trials=args.trials,
            proxy_phase1_epochs=args.proxy_phase1_epochs,
            proxy_phase2_epochs=args.proxy_phase2_epochs,
            proxy_patience=args.proxy_patience,
            method=args.search_method
        )

        best_row = df.iloc[0].to_dict()
        best_tuned_cfg = best_cfg

        search_meta.update({
            "trials_csv": trials_csv,
            "proxy": {
                "phase1_epochs": int(args.proxy_phase1_epochs),
                "phase2_epochs": int(args.proxy_phase2_epochs),
                "patience": int(args.proxy_patience),
            },
            "domain": dom,
            "selection_rule": "pick config with lowest proxy best_val_loss (tie-break by best_val_acc)",
            "best_proxy": {
                "val_loss": float(best_row["best_val_loss"]),
                "val_acc": float(best_row["best_val_acc"]),
                "cfg": best_cfg
            }
        })

        save_json(best_tuned_path, best_tuned_cfg)
        print(f"✅ Saved best tuned cfg -> {best_tuned_path}")
        print("✅ Selected -> best_cfg (pure search)")

    # ---- Train FINAL dengan best_tuned_cfg ----
    # Ini training utama (lebih panjang) menggunakan lr1/lr2/unfreeze terbaik hasil tuning.
    model, base_model = build_model_from_cfg(best_tuned_cfg, num_classes=train_gen.num_classes, img_size=IMG_SIZE)

    best_weights_path  = os.path.join(MODEL_DIR, "best_classification_model.weights.h5")
    last_weights_path  = os.path.join(MODEL_DIR, "final_classification_model.weights.h5")

    # Save best weights berdasarkan val_loss agar generalisasi lebih baik.
    ckpt_best_val_loss = ModelCheckpoint(
        filepath=best_weights_path, monitor='val_loss',
        save_best_only=True, save_weights_only=True, verbose=1
    )

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    steps_per_epoch = int(np.ceil(train_gen.samples / BATCH_SIZE))

    # Phase 1 (FINAL): latih head dulu, backbone frozen (stabil).
    base_model.trainable = False
    lr1 = float(best_tuned_cfg["schedule"]["lr_phase1"])
    opt1 = Adam(learning_rate=lr1, clipnorm=1.0)
    model.compile(optimizer=opt1, loss=loss_fn, metrics=['accuracy'])

    earlystop_cb1 = EarlyStopping(monitor='val_loss', patience=6, min_delta=1e-5,
                                  restore_best_weights=True, verbose=1)
    reduce_lr_cb1 = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=2, min_lr=1e-7, verbose=1)

    print("=== Phase 1: Training top layers (base frozen) ===")
    hist1 = model.fit(
        train_gen, validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS_PHASE1,
        callbacks=[ckpt_best_val_loss, earlystop_cb1, reduce_lr_cb1],
        workers=1, use_multiprocessing=False
    )

    # Phase 2 (FINAL): fine-tune sebagian layer terakhir backbone, BN dibekukan.
    print("=== Phase 2: Fine-tuning last layers of EfficientNetB4 (BN frozen) ===")
    base_model.trainable = True

    unfreeze_last = int(best_tuned_cfg["schedule"]["unfreeze_last_n_layers"])
    freeze_until = max(0, len(base_model.layers) - unfreeze_last)

    # Bekukan layer awal (fitur umum), buka layer akhir (adaptasi domain).
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True

    # Bekukan BN untuk stabilitas batch kecil.
    freeze_batchnorm_layers(base_model)

    lr2 = float(best_tuned_cfg["schedule"]["lr_phase2"])
    opt2 = Adam(learning_rate=lr2, clipnorm=1.0)
    model.compile(optimizer=opt2, loss=loss_fn, metrics=['accuracy'])

    earlystop_cb2 = EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-5,
                                  restore_best_weights=True, verbose=1)
    reduce_lr_cb2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

    hist2 = model.fit(
        train_gen, validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        initial_epoch=EPOCHS_PHASE1, epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2,
        callbacks=[ckpt_best_val_loss, earlystop_cb2, reduce_lr_cb2],
        workers=1, use_multiprocessing=False
    )

    # ===== Save & Plot Training Curves (buat TA / paper) =====
    merged_hist = merge_histories([hist1, hist2])

    save_history(merged_hist, out_dir=MODEL_DIR, prefix="classification")
    plot_training_curves(merged_hist, out_dir=MODEL_DIR, prefix="classification")


    # Simpan last epoch weights (opsional) + best-val-loss weights (utama).
    model.save_weights(last_weights_path)
    print(f"✅ Saved last-epoch weights -> {last_weights_path}")
    print(f"✅ Saved best-val-loss weights -> {best_weights_path}")

    # Simpan spec untuk dokumentasi.
    save_spec(best_weights_path, last_weights_path, class_indices_path, best_tuned_cfg, search_meta)
    K.clear_session()


# ========= Eval =========
def eval_mode(args):
    K.clear_session()

    # Load mapping index->label untuk laporan.
    class_indices_path = os.path.join(MODEL_DIR, "classification_class_indices.json")
    idx_to_class = None
    if os.path.exists(class_indices_path):
        _, idx_to_class = load_class_indices(class_indices_path)

    # Wajib: load cfg hasil tuning (tidak ada fallback).
    best_tuned_path = os.path.join(MODEL_DIR, "best_tuned_cfg.json")
    if not os.path.exists(best_tuned_path):
        raise RuntimeError("best_tuned_cfg.json not found. Run training/search first to generate it.")
    with open(best_tuned_path, "r") as f:
        cfg = json.load(f)

    # Untuk eval, set float32 agar stabil (hindari isu numerik saat menghitung log loss manual).
    prev_policy = mixed_precision.global_policy()
    mixed_precision.set_global_policy('float32')
    try:
        model, _ = build_model_from_cfg(cfg, num_classes=NUM_CLASSES, img_size=IMG_SIZE, activation="softmax", dtype_last="float32")
        model.load_weights(args.weights)

        gen = make_eval_datagen().flow_from_directory(
            VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
            class_mode='categorical', shuffle=False, seed=GLOBAL_SEED
        )
        filepaths = list(getattr(gen, "filepaths", []))

        # Manual loop predict untuk mengumpulkan probs & labels.
        n_steps = int(np.ceil(gen.n / gen.batch_size))
        all_probs, all_labels = [], []
        for _ in range(n_steps):
            xb, yb = next(gen)
            p = model.predict(xb, verbose=0)
            all_probs.append(p)
            all_labels.append(yb)

        probs = np.concatenate(all_probs, axis=0)[:gen.n]
        labels = np.concatenate(all_labels, axis=0)[:gen.n]

        # Cross-entropy manual untuk val_loss (sesuai softmax).
        eps = 1e-12
        val_loss = float((-np.sum(labels * np.log(np.clip(probs, eps, 1.0)), axis=1)).mean())
        preds = probs.argmax(axis=1)
        true  = labels.argmax(axis=1)
        val_acc = float((preds == true).mean())

        print("\n=== Evaluation on VAL set (no TTA, no EMA, no temperature scaling) ===")
        print(f"📊 Val Loss: {val_loss:.4f}")
        print(f"✅ Val Accuracy: {val_acc:.4f}")

        cm, rep = confusion_and_report(true, preds, probs.shape[1], idx_to_class)

        # Output files.
        base = args.out_csv[:-4] if args.out_csv.endswith(".csv") else args.out_csv
        report_csv = f"{base}_report.csv"
        cm_csv     = f"{base}_confusion.csv"
        pred_csv   = f"{base}_predictions.csv"

        save_report_csv(rep, idx_to_class, report_csv)
        save_confusion_csv(cm, idx_to_class, cm_csv)

        plot_confusion_png(cm, idx_to_class, f"{base}_confusion.png", normalize=False)
        plot_confusion_png(cm, idx_to_class, f"{base}_confusion_normalized.png", normalize=True)

        # Per-sample prediction CSV untuk analisis error.
        conf = np.max(probs, axis=1)
        rows = []
        for i in range(len(true)):
            t = int(true[i]); p = int(preds[i])
            rows.append({
                "filepath": filepaths[i] if i < len(filepaths) else "",
                "true_idx": t,
                "pred_idx": p,
                "true_label": idx_to_class[t] if idx_to_class else str(t),
                "pred_label": idx_to_class[p] if idx_to_class else str(p),
                "confidence": float(conf[i]),
                "correct": bool(t == p)
            })
        pd.DataFrame(rows).to_csv(pred_csv, index=False)

        # Grid visual untuk paper: stratified true vs pred.
        if filepaths and len(filepaths) >= len(true):
            save_true_vs_pred_grid(
                filepaths=filepaths,
                y_true=true,
                y_pred=preds,
                probs=probs,
                idx_to_class=idx_to_class,
                out_png=f"{base}_true_vs_pred_grid.png",
                max_items=int(args.grid_n),
                samples_per_class=args.grid_per_class,
                seed=GLOBAL_SEED
            )

        print(f"\n📝 Saved report CSV      -> {report_csv}")
        print(f"🧩 Saved confusion CSV   -> {cm_csv}")
        print(f"🧾 Saved predictions CSV -> {pred_csv}")
        print(f"🖼️  Saved confusion PNG  -> {base}_confusion.png")
        print(f"🖼️  Saved norm CM PNG     -> {base}_confusion_normalized.png")
        if filepaths:
            print(f"🧷 Saved True-vs-Pred PNG -> {base}_true_vs_pred_grid.png")

    finally:
        mixed_precision.set_global_policy(prev_policy)
        K.clear_session()


# ========= Predict =========


# ========= Entrypoint =========
if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train_mode(args)
    elif args.mode == "eval":
        eval_mode(args)
    elif args.mode == "predict":
        predict_mode(args)


# ============================================================
# Tools entrypoints (CLI) — extracted from exe_classification_model.py
# Provides: tune / train / eval
# ============================================================

def tune(args):
    """Hyperparameter+head-architecture search (proxy training), saves best_tuned_cfg.json + trials CSV."""
    K.clear_session()

    train_datagen = make_train_datagen()
    eval_datagen  = make_eval_datagen()

    train_gen = train_datagen.flow_from_directory(
        args.train_dir, target_size=IMG_SIZE, batch_size=args.batch_size,
        class_mode='categorical', shuffle=True, seed=GLOBAL_SEED
    )
    val_gen = eval_datagen.flow_from_directory(
        args.val_dir, target_size=IMG_SIZE, batch_size=args.batch_size,
        class_mode='categorical', shuffle=False, seed=GLOBAL_SEED
    )

    os.makedirs(args.model_dir, exist_ok=True)

    # Save class indices mapping (important for consistent eval later)
    class_indices_path = os.path.join(args.model_dir, "classification_class_indices.json")
    with open(class_indices_path, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    dom = build_search_domain_from_args(args)

    best_cfg, df, trials_csv = tune_cfg_cand_search(
        train_gen, val_gen,
        dom=dom,
        trials=args.trials,
        proxy_phase1_epochs=args.proxy_phase1_epochs,
        proxy_phase2_epochs=args.proxy_phase2_epochs,
        proxy_patience=args.proxy_patience,
        method=args.search_method
    )

    best_tuned_path = os.path.join(args.model_dir, "best_tuned_cfg.json")
    save_json(best_cfg, best_tuned_path)
    print(f"✅ Saved best tuned cfg -> {best_tuned_path}")
    print(f"✅ Trials summary -> {trials_csv}")

    # also save run spec (reproducibility)
    spec = {
        "mode": "tune",
        "search_method": args.search_method,
        "trials": args.trials,
        "proxy_phase1_epochs": args.proxy_phase1_epochs,
        "proxy_phase2_epochs": args.proxy_phase2_epochs,
        "proxy_patience": args.proxy_patience,
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "batch_size": args.batch_size,
        "img_size": list(IMG_SIZE),
        "seed": GLOBAL_SEED,
    }
    save_spec(spec, os.path.join(args.model_dir, "tune_spec.json"))


def train(args):
    """Final training (2-phase) using best_tuned_cfg.json."""
    # If user insists cfg must exist:
    best_tuned_path = os.path.join(args.model_dir, "best_tuned_cfg.json")
    if args.require_tuned and (not os.path.exists(best_tuned_path)):
        raise FileNotFoundError(
            f"Missing {best_tuned_path}. Run `tune` first or pass --require_tuned 0 to auto-search."
        )

    # Reuse existing train_mode logic (it will tune if missing and require_tuned=0)
    args.force_search = (not args.require_tuned)
    train_mode(args)


def evaluate(args):
    """Evaluate saved model on validation set and export metrics/artifacts."""
    eval_mode(args)


def parse_args_cli():
    p = argparse.ArgumentParser(
        description="EfficientNetB4 classification tools: tune/train/eval"
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(io):
        io.add_argument("--train_dir", default=TRAIN_DIR, help="Path to training folder (ImageDataGenerator flow_from_directory)")
        io.add_argument("--val_dir", default=VAL_DIR, help="Path to validation folder (flow_from_directory)")
        io.add_argument("--model_dir", default=MODEL_DIR, help="Output folder for models and artifacts")
        io.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        io.add_argument("--seed", type=int, default=GLOBAL_SEED)

    # --- tune ---
    t = sub.add_parser("tune", help="Search best head/hparams via proxy training")
    add_common(t)
    t.add_argument("--search_method", choices=["random", "grid"], default="random")
    t.add_argument("--trials", type=int, default=12)
    t.add_argument("--proxy_phase1_epochs", type=int, default=2)
    t.add_argument("--proxy_phase2_epochs", type=int, default=4)
    t.add_argument("--proxy_patience", type=int, default=2)
    # search domain knobs (forwarded to build_search_domain_from_args)
    t.add_argument("--head_pool", nargs="+", default=None)
    t.add_argument("--head_dropout1", nargs="+", default=None)
    t.add_argument("--head_dropout2", nargs="+", default=None)
    t.add_argument("--head_dense_units", nargs="+", default=None)
    t.add_argument("--head_dense2_units", nargs="+", default=None)
    t.add_argument("--head_use_bn", nargs="+", default=None)
    t.add_argument("--lr1", nargs="+", default=None)
    t.add_argument("--lr2", nargs="+", default=None)
    t.add_argument("--unfreeze_layers", nargs="+", default=None)

    # --- train ---
    tr = sub.add_parser("train", help="Train final model using best_tuned_cfg.json")
    add_common(tr)
    tr.add_argument("--require_tuned", type=int, default=1, help="1: require best_tuned_cfg.json exists; 0: auto-search if missing")
    tr.add_argument("--epochs_phase1", type=int, default=EPOCHS_PHASE1)
    tr.add_argument("--epochs_phase2", type=int, default=EPOCHS_PHASE2)
    tr.add_argument("--patience", type=int, default=6)

    # --- eval ---
    ev = sub.add_parser("eval", help="Evaluate trained model on validation")
    add_common(ev)
    ev.add_argument("--model_path", default=None, help="Optional explicit model file; default uses best_model.keras in model_dir")
    ev.add_argument("--topk", type=int, default=3)

    return p.parse_args()


def main():
    args = parse_args_cli()

    # keep backward-compat for functions expecting globals/constants
    global TRAIN_DIR, VAL_DIR, MODEL_DIR, BATCH_SIZE, GLOBAL_SEED, EPOCHS_PHASE1, EPOCHS_PHASE2
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    MODEL_DIR = args.model_dir
    BATCH_SIZE = args.batch_size
    GLOBAL_SEED = args.seed
    if getattr(args, "epochs_phase1", None) is not None:
        EPOCHS_PHASE1 = args.epochs_phase1
    if getattr(args, "epochs_phase2", None) is not None:
        EPOCHS_PHASE2 = args.epochs_phase2

    if args.cmd == "tune":
        tune(args)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
