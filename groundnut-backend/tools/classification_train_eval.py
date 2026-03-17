# tools/classification_train_eval.py
# EfficientNetB4 Classification with Tune/Train/Eval (simplified structure)
import os, json, argparse, random, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ( Dense, GlobalAveragePooling2D, GlobalMaxPooling2D,
    Dropout, BatchNormalization, Concatenate )
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras import mixed_precision, backend as K

BASE_DIR = "datasets/processed/classification_dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train_balanced")
VAL_DIR = os.path.join(BASE_DIR, "val")
MODEL_DIR = "models/classification"

IMG_SIZE = (380, 380)
NUM_CLASSES = 5
BATCH = 16
SEED = 42
E1 = 3
E2 = 45

def setup_runtime(seed: int):
    mixed_precision.set_global_policy("mixed_float16")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ------------------ IO Helpers & Data ------------------
def jsave(path: str, obj: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def jload(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_class_indices(path: str):
    class_indices = jload(path)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return class_indices, idx_to_class

def make_gens(train_dir: str, val_dir: str, batch: int, seed: int):
    train_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=batch, class_mode="categorical",
        shuffle=True, seed=seed
    )
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
        val_dir, target_size=IMG_SIZE, batch_size=batch, class_mode="categorical",
        shuffle=False, seed=seed
    )
    return train_gen, val_gen

# ------------------ Model ------------------
def freeze_bn(model):
    for layer in model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

def apply_head(x, head_cfg: dict, num_classes: int, activation="softmax", dtype_last="float32"):
    head_type = str(head_cfg["head_type"])
    drop1 = float(head_cfg["drop1"])
    drop2 = float(head_cfg["drop2"])
    dense_units = int(head_cfg["dense_units"])
    dense_l2 = float(head_cfg.get("dense_l2", 0.0))
    reg = l2(dense_l2) if dense_l2 > 0 else None

    if head_type == "gap_gmp":
        gap = GlobalAveragePooling2D(name="gap")(x)
        gmp = GlobalMaxPooling2D(name="gmp")(x)
        x = Concatenate(name="gap_gmp_concat")([gap, gmp])
    else:
        x = GlobalAveragePooling2D(name="gap")(x)

    act = "swish" if head_type in ("swish", "mlp2") else "relu"

    x = Dropout(drop1, name="drop1")(x)
    x = Dense(dense_units, activation=act, kernel_regularizer=reg, name="dense")(x)
    x = Dropout(drop2, name="drop2")(x)

    if head_type == "mlp2":
        dense_units2 = int(head_cfg.get("dense_units2", max(64, dense_units // 2)))
        drop3 = float(head_cfg.get("drop3", 0.2))
        x = Dense(dense_units2, activation=act, kernel_regularizer=reg, name="dense2")(x)
        x = Dropout(drop3, name="drop3")(x)

    out = Dense(num_classes, activation=activation, dtype=dtype_last, name="predictions")(x)
    return out

def build_model(cfg: dict, num_classes: int, activation="softmax", dtype_last="float32"):
    base = EfficientNetB4(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    out = apply_head(base.output, cfg["head"], num_classes, activation=activation, dtype_last=dtype_last)
    return Model(inputs=base.input, outputs=out, name="EffNetB4_Classifier"), base

# ------------------ Metrics + Reports ------------------
def confusion_and_report(true_idx, pred_idx, num_classes, idx_to_class=None):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_idx, pred_idx):
        cm[int(t), int(p)] += 1

    eps = 1e-12
    support = cm.sum(axis=1).astype(np.int64)
    tp = np.diag(cm).astype(np.float64)

    # Per-class metrics
    precision = tp / np.maximum(cm.sum(axis=0).astype(np.float64), eps)
    recall    = tp / np.maximum(cm.sum(axis=1).astype(np.float64), eps)
    f1        = 2 * precision * recall / np.maximum(precision + recall, eps)

    macro_a = float(tp.sum() / np.maximum(cm.sum(), 1))
    macro_p  = float(np.mean(precision))
    macro_r  = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    names = [idx_to_class[i] if idx_to_class else str(i) for i in range(num_classes)]

    print("\nPer-class metrics:")
    for i, name in enumerate(names):
        print(
            f"  {name:>28s} | "
            f"P:{precision[i]:.3f}  R:{recall[i]:.3f}  F1:{f1[i]:.3f}  Support:{int(support[i])}"
        )

    print(
        f"\nMacroA:{macro_a:.4f}  "
        f"MacroP:{macro_p:.4f}  MacroR:{macro_r:.4f}  MacroF1:{macro_f1:.4f}"
    )

    rep = {
        "precision": precision, "recall": recall, "f1": f1,
        "macro_a": macro_a, "macro_p": macro_p, "macro_r": macro_r, "macro_f1": macro_f1, "support": support,
    }
    return cm, rep

def save_report_csv(rep, idx_to_class, path):
    rows = []
    for i in range(len(rep["precision"])):
        name = idx_to_class[i] if idx_to_class else str(i)
        rows.append({
            "class": name,
            "precision": float(rep["precision"][i]),
            "recall": float(rep["recall"][i]),
            "f1": float(rep["f1"][i]),
            "support": int(rep["support"][i]),
        })
    rows.append({
        "class": "OVERALL",
        "accuracy": float(rep["macro_a"]),
        "precision": float(rep["macro_p"]),
        "recall": float(rep["macro_r"]),
        "f1": float(rep["macro_f1"]),
        "support": int(np.sum(rep["support"])),
    })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)

def save_confusion_csv(cm, idx_to_class, path):
    labels = [idx_to_class[i] for i in range(cm.shape[0])] if idx_to_class else [str(i) for i in range(cm.shape[0])]
    df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=True)

def plot_confusion_png(cm, idx_to_class, out_png, normalize=False):
    labels = [idx_to_class[i] for i in range(cm.shape[0])] if idx_to_class else [str(i) for i in range(cm.shape[0])]
    mat = cm.astype(np.float64)
    if normalize:
        mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1e-12)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, interpolation="nearest", cmap="Blues", vmin=0)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized Count" if normalize else "Count", rotation=270, labelpad=20)

    ax.set_title("Confusion Matrix (Normalized)" if normalize else "Confusion Matrix", fontsize=14, weight="bold", pad=20)
    ax.set_xlabel("Predicted", fontsize=12, weight="bold")
    ax.set_ylabel("True", fontsize=12, weight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    thresh = mat.max() / 2.0 if mat.size else 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = f"{v:.2f}" if normalize else str(int(v))
            text_color = "white" if v > thresh else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=text_color, weight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_training_curves(history: dict, out_dir: str, prefix="classification"):
    os.makedirs(out_dir, exist_ok=True)
    if not history:
        print("[WARN] Empty history dict. Skip plotting.")
        return

    n_epochs = None
    for _, v in history.items():
        if isinstance(v, (list, tuple)) and len(v) > 0:
            n_epochs = len(v)
            break
    if not n_epochs:
        print("[WARN] History has no epoch data. Skip plotting.")
        return

    epochs = np.arange(1, n_epochs + 1)

    if "loss" in history:
        plt.figure()
        plt.plot(epochs, history.get("loss", []), label="train_loss")
        if "val_loss" in history:
            plt.plot(epochs, history.get("val_loss", []), label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{prefix}_curve_loss.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

    acc_key = "accuracy" if "accuracy" in history else ("acc" if "acc" in history else None)
    if acc_key:
        val_acc_key = "val_accuracy" if "val_accuracy" in history else ("val_acc" if "val_acc" in history else None)
        plt.figure()
        plt.plot(epochs, history.get(acc_key, []), label="train_accuracy")
        if val_acc_key:
            plt.plot(epochs, history.get(val_acc_key, []), label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{prefix}_curve_accuracy.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

def save_history(history: dict, out_dir: str, prefix="classification"):
    os.makedirs(out_dir, exist_ok=True)
    safe = {}
    for k, v in history.items():
        if isinstance(v, (list, tuple)):
            safe[k] = [float(x) for x in v]
        else:
            safe[k] = float(v)

    jpath = os.path.join(out_dir, f"{prefix}_train_history.json")
    cpath = os.path.join(out_dir, f"{prefix}_train_history.csv")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2)
    pd.DataFrame(safe).to_csv(cpath, index=False)

def save_true_vs_pred_grid(filepaths, y_true, y_pred, probs, idx_to_class, out_png,
                           max_items=16, samples_per_class=None, seed=42):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    probs = np.asarray(probs)
    n = min(len(y_true), len(y_pred), probs.shape[0], len(filepaths))
    if n <= 0:
        print("No samples to plot for True vs Pred grid.")
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

    chosen, chosen_set = [], set()

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
        print("No samples to plot for True vs Pred grid.")
        return []

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    base_no_ext, ext = os.path.splitext(out_png)
    if not ext:
        ext = ".png"

    # One slide/page = 2x2 grid (4 images) for readability in presentations.
    items_per_page = 4
    cols, rows = 2, 2
    wrap_width = 24
    saved_pages = []
    total_pages = int(np.ceil(len(chosen) / items_per_page))

    for page_idx in range(total_pages):
        start = page_idx * items_per_page
        end = min(start + items_per_page, len(chosen))
        page_items = chosen[start:end]

        fig, axes = plt.subplots(rows, cols, figsize=(12, 12), squeeze=False)
        fig.patch.set_facecolor("white")
        for ax in axes.flat:
            ax.axis("off")

        for k, idx in enumerate(page_items):
            row = k // cols
            col = k % cols
            ax = axes[row, col]

            img = tf.keras.utils.load_img(filepaths[idx], target_size=IMG_SIZE)
            ax.imshow(img)
            ax.axis("off")

            tname = idx_to_class[int(y_true[idx])] if idx_to_class else str(int(y_true[idx]))
            pname = idx_to_class[int(y_pred[idx])] if idx_to_class else str(int(y_pred[idx]))

            # Keep full labels, but wrap to multiple lines for slide readability.
            t_wrapped = textwrap.fill(tname.replace("_", " "), width=wrap_width, break_long_words=False, break_on_hyphens=False)
            p_wrapped = textwrap.fill(pname.replace("_", " "), width=wrap_width, break_long_words=False, break_on_hyphens=False)
            is_correct = int(y_true[idx]) == int(y_pred[idx])
            status = "Correct" if is_correct else "Wrong"
            status_color = "#1f7a1f" if is_correct else "#b22222"

            label_text = (
                f"True: {t_wrapped}\n"
                f"Pred: {p_wrapped}\n"
                f"Conf: {conf[idx]:.2f}  |  {status}"
            )
            ax.text(
                0.5, -0.08, label_text,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=11,
                color=status_color,
                bbox={"facecolor": "white", "edgecolor": "#dddddd", "boxstyle": "round,pad=0.35"}
            )

        fig.suptitle(
            f"True Label vs Predicted Label (Page {page_idx + 1}/{total_pages})",
            fontsize=16, weight="bold", y=0.995
        )
        plt.tight_layout(rect=[0, 0, 1, 0.975])

        page_path = f"{base_no_ext}_page_{page_idx + 1:02d}{ext}"
        plt.savefig(page_path, dpi=220, facecolor="white")
        plt.close(fig)
        saved_pages.append(page_path)

    return saved_pages

# ------------------ Search (Tune) ------------------
def cfg_key(cfg: dict):
    h, s = cfg["head"], cfg["schedule"]
    return (
        str(h["head_type"]),
        int(h["dense_units"]), float(h["drop1"]), float(h["drop2"]), float(h.get("dense_l2", 0.0)),
        int(h.get("dense_units2", 0)), float(h.get("drop3", 0.0)),
        float(s["lr_phase1"]), float(s["lr_phase2"]), int(s["unfreeze_last_n_layers"])
    )

def build_domain(args) -> dict:
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

def sample_cfg(dom: dict, rng) -> dict:
    return {
        "head": {
            "head_type": str(rng.choice(dom["head_type"])),
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

def gen_candidates(dom: dict, method: str, trials: int, seed: int):
    if method == "random":
        rng = np.random.default_rng(seed)
        cands = [sample_cfg(dom, rng) for _ in range(int(trials))]

    uniq = {}
    for c in cands:
        uniq[cfg_key(c)] = c
    return list(uniq.values())

def proxy_eval(train_gen, val_gen, cfg: dict, batch: int, p1: int, p2: int, patience: int):
    K.clear_session()

    model, base = build_model(cfg, num_classes=train_gen.num_classes)
    steps = int(np.ceil(train_gen.samples / batch))
    val_steps = int(np.ceil(val_gen.samples / val_gen.batch_size))
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    base.trainable = False
    model.compile(optimizer=Adam(float(cfg["schedule"]["lr_phase1"]), clipnorm=1.0), loss=loss_fn, metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, steps_per_epoch=steps, validation_steps=val_steps,
              epochs=int(p1), verbose=0, workers=1, use_multiprocessing=False)

    base.trainable = True
    unfreeze_last = int(cfg["schedule"]["unfreeze_last_n_layers"])
    freeze_until = max(0, len(base.layers) - unfreeze_last)
    for layer in base.layers[:freeze_until]:
        layer.trainable = False
    for layer in base.layers[freeze_until:]:
        layer.trainable = True
    freeze_bn(base)

    model.compile(optimizer=Adam(float(cfg["schedule"]["lr_phase2"]), clipnorm=1.0), loss=loss_fn, metrics=["accuracy"])
    es = EarlyStopping(monitor="val_loss", patience=int(patience), min_delta=1e-5, restore_best_weights=True, verbose=0)
    hist = model.fit(train_gen, validation_data=val_gen, steps_per_epoch=steps, validation_steps=val_steps,
                     epochs=int(p2), callbacks=[es], verbose=0, workers=1, use_multiprocessing=False)

    h = hist.history
    best_vloss = float(np.min(h.get("val_loss", [np.inf])))
    best_vacc = float(np.max(h.get("val_accuracy", [0.0])))

    K.clear_session()
    return best_vloss, best_vacc

def run_tune(args):
    os.makedirs(args.model_dir, exist_ok=True)
    train_gen, val_gen = make_gens(args.train_dir, args.val_dir, args.batch_size, args.seed)
    class_idx_path = os.path.join(args.model_dir, "classification_class_indices.json")
    jsave(class_idx_path, train_gen.class_indices)
    dom = build_domain(args)
    cands = gen_candidates(dom, args.search_method, args.trials, args.seed)

    print(f"\nTune candidates ({args.search_method}): {len(cands)}")
    if args.search_method == "grid" and len(cands) > 2000:
        print("[WARN] Grid sangat besar, pertimbangkan kecilkan domain list.")

    rows = []
    for i, cfg in enumerate(cands, start=1):
        print(f"  Trial {i}/{len(cands)} | head_type={cfg['head']['head_type']}")
        vloss, vacc = proxy_eval(train_gen, val_gen, cfg, args.batch_size, args.proxy_phase1_epochs, args.proxy_phase2_epochs, args.proxy_patience)
        rows.append({
            **cfg["head"],
            **cfg["schedule"],
            "best_val_loss": vloss,
            "best_val_acc": vacc,
        })

    df = pd.DataFrame(rows).sort_values(["best_val_loss", "best_val_acc"], ascending=[True, False])
    trials_csv = os.path.join(args.model_dir, "tune_cfg_cand_trials.csv")
    df.to_csv(trials_csv, index=False)
    best_row = df.iloc[0].to_dict()
    best_cfg = {
        "head": {
            "head_type": str(best_row["head_type"]),
            "dense_units": int(best_row["dense_units"]),
            "drop1": float(best_row["drop1"]),
            "drop2": float(best_row["drop2"]),
            "dense_l2": float(best_row["dense_l2"]),
            "dense_units2": int(best_row.get("dense_units2", 128)),
            "drop3": float(best_row.get("drop3", 0.2)),
        },
        "schedule": {
            "lr_phase1": float(best_row["lr_phase1"]),
            "lr_phase2": float(best_row["lr_phase2"]),
            "unfreeze_last_n_layers": int(best_row["unfreeze_last_n_layers"]),
        }
    }

    best_cfg_path = os.path.join(args.model_dir, "best_tuned_cfg.json")
    jsave(best_cfg_path, best_cfg)
    tune_spec = {
        "cmd": "tune",
        "search_method": args.search_method,
        "trials": int(args.trials),
        "proxy": {
            "phase1_epochs": int(args.proxy_phase1_epochs),
            "phase2_epochs": int(args.proxy_phase2_epochs),
            "patience": int(args.proxy_patience),
        },
        "domain": dom,
        "trials_csv": trials_csv,
        "best_proxy": {
            "val_loss": float(best_row["best_val_loss"]),
            "val_acc": float(best_row["best_val_acc"]),
            "cfg": best_cfg
        }
    }
    jsave(os.path.join(args.model_dir, "tune_spec.json"), tune_spec)
    print(f"\nSaved trials -> {trials_csv}")
    print(f"Saved best cfg -> {best_cfg_path}")

# ------------------ Train (final) ------------------
def run_train(args):
    os.makedirs(args.model_dir, exist_ok=True)
    train_gen, val_gen = make_gens(args.train_dir, args.val_dir, args.batch_size, args.seed)
    class_idx_path = os.path.join(args.model_dir, "classification_class_indices.json")
    if not os.path.exists(class_idx_path):
        jsave(class_idx_path, train_gen.class_indices)

    best_cfg_path = os.path.join(args.model_dir, "best_tuned_cfg.json")
    if not os.path.exists(best_cfg_path):
        if args.require_tuned:
            raise FileNotFoundError(f"Missing {best_cfg_path}. Jalankan tune dulu atau set --require_tuned 0.")
        run_tune(args)

    cfg = jload(best_cfg_path)
    model, base = build_model(cfg, num_classes=train_gen.num_classes)
    weights_path = os.path.join(args.model_dir, "best_classification_model.weights.h5")
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    steps = int(np.ceil(train_gen.samples / args.batch_size))
    ckpt = ModelCheckpoint(weights_path, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)

    base.trainable = False
    model.compile(
        optimizer=Adam(float(cfg["schedule"]["lr_phase1"]), clipnorm=1.0),
        loss=loss_fn, metrics=["accuracy"]
    )
    es1 = EarlyStopping(monitor="val_loss", patience=6, min_delta=1e-5, restore_best_weights=True, verbose=1)
    rl1 = ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=2, min_lr=1e-7, verbose=1)

    print("=== Phase 1: Training top layers (base frozen) ===")
    h1 = model.fit(
        train_gen, validation_data=val_gen, steps_per_epoch=steps,
        epochs=args.epochs_phase1, callbacks=[ckpt, es1, rl1],
        workers=1, use_multiprocessing=False
    )

    print("=== Phase 2: Fine-tuning last layers (BN frozen) ===")
    base.trainable = True
    unfreeze_last = int(cfg["schedule"]["unfreeze_last_n_layers"])
    freeze_until = max(0, len(base.layers) - unfreeze_last)
    for layer in base.layers[:freeze_until]:
        layer.trainable = False
    for layer in base.layers[freeze_until:]:
        layer.trainable = True
    freeze_bn(base)
    model.compile(
        optimizer=Adam(float(cfg["schedule"]["lr_phase2"]), clipnorm=1.0),
        loss=loss_fn, metrics=["accuracy"]
    )
    es2 = EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-5, restore_best_weights=True, verbose=1)
    rl2 = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    h2 = model.fit(
        train_gen, validation_data=val_gen, steps_per_epoch=steps,
        initial_epoch=args.epochs_phase1,
        epochs=args.epochs_phase1 + args.epochs_phase2,
        callbacks=[ckpt, es2, rl2],
        workers=1, use_multiprocessing=False
    )

    hist = {}
    for hh in [h1, h2]:
        for k, v in hh.history.items():
            hist.setdefault(k, [])
            hist[k].extend(list(v))

    save_history(hist, args.model_dir, prefix="classification")
    plot_training_curves(hist, args.model_dir, prefix="classification")
    train_spec = {
        "cmd": "train",
        "img_size": list(IMG_SIZE),
        "batch_size": int(args.batch_size),
        "epochs_phase1": int(args.epochs_phase1),
        "epochs_phase2": int(args.epochs_phase2),
        "weights": weights_path,
        "class_indices": class_idx_path,
        "best_tuned_cfg": cfg,
    }
    jsave(os.path.join(args.model_dir, "classification_model_spec.json"), train_spec)
    print(f"Saved weights -> {weights_path}")
    K.clear_session()

# ------------------ Eval ------------------
def run_eval(args):
    os.makedirs(args.model_dir, exist_ok=True)
    class_idx_path = os.path.join(args.model_dir, "classification_class_indices.json")
    idx_to_class = None
    if os.path.exists(class_idx_path):
        _, idx_to_class = load_class_indices(class_idx_path)

    cfg_path = os.path.join(args.model_dir, "best_tuned_cfg.json")
    if not os.path.exists(cfg_path):
        raise RuntimeError("best_tuned_cfg.json tidak ditemukan. Jalankan tune/train dulu.")

    cfg = jload(cfg_path)
    prev_policy = mixed_precision.global_policy()
    mixed_precision.set_global_policy("float32")
    try:
        model, _ = build_model(cfg, num_classes=NUM_CLASSES, activation="softmax", dtype_last="float32")
        model.load_weights(args.weights)

        gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
            args.val_dir, target_size=IMG_SIZE, batch_size=args.batch_size, class_mode="categorical",
            shuffle=False, seed=args.seed
        )
        filepaths = list(getattr(gen, "filepaths", []))
        n_steps = int(np.ceil(gen.n / gen.batch_size))
        all_probs, all_labels = [], []
        for _ in range(n_steps):
            xb, yb = next(gen)
            p = model.predict(xb, verbose=0)
            all_probs.append(p)
            all_labels.append(yb)

        probs = np.concatenate(all_probs, axis=0)[:gen.n]
        labels = np.concatenate(all_labels, axis=0)[:gen.n]
        eps = 1e-12
        val_loss = float((-np.sum(labels * np.log(np.clip(probs, eps, 1.0)), axis=1)).mean())
        preds = probs.argmax(axis=1)
        true = labels.argmax(axis=1)
        val_acc = float((preds == true).mean())

        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")

        cm, rep = confusion_and_report(true, preds, probs.shape[1], idx_to_class)

        base = args.out_prefix
        report_csv = f"{base}_report.csv"
        cm_csv = f"{base}_confusion.csv"
        pred_csv = f"{base}_predictions.csv"
        save_report_csv(rep, idx_to_class, report_csv)
        save_confusion_csv(cm, idx_to_class, cm_csv)
        plot_confusion_png(cm, idx_to_class, f"{base}_confusion.png", normalize=False)
        plot_confusion_png(cm, idx_to_class, f"{base}_confusion_normalized.png", normalize=True)
        conf = np.max(probs, axis=1)
        rows = []
        for i in range(len(true)):
            t, p = int(true[i]), int(preds[i])
            rows.append({
                "filepath": filepaths[i] if i < len(filepaths) else "",
                "true_idx": t,
                "pred_idx": p,
                "true_label": idx_to_class[t] if idx_to_class else str(t),
                "pred_label": idx_to_class[p] if idx_to_class else str(p),
                "confidence": float(conf[i]),
                "correct": bool(t == p),
            })
        pd.DataFrame(rows).to_csv(pred_csv, index=False)

        if filepaths and len(filepaths) >= len(true):
            grid_pages = save_true_vs_pred_grid(
                filepaths=filepaths, y_true=true, y_pred=preds, probs=probs,
                idx_to_class=idx_to_class, out_png=f"{base}_true_vs_pred_grid.png",
                max_items=int(args.grid_n), samples_per_class=args.grid_per_class, seed=args.seed
            )

        print(f"\nSaved report CSV       -> {report_csv}")
        print(f"Saved confusion CSV    -> {cm_csv}")
        print(f"Saved predictions CSV  -> {pred_csv}")
        print(f"Saved confusion PNG    -> {base}_confusion.png")
        print(f"Saved norm CM PNG      -> {base}_confusion_normalized.png")
        if filepaths:
            if grid_pages:
                print(f"Saved True-vs-Pred PNG pages -> {base}_true_vs_pred_grid_page_XX.png ({len(grid_pages)} pages)")
            else:
                print("Saved True-vs-Pred PNG pages -> none")

    finally:
        mixed_precision.set_global_policy(prev_policy)
        K.clear_session()

# ------------------ CLI ------------------
def parse_args():
    p = argparse.ArgumentParser(description="EfficientNetB4: tune/train/eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--train_dir", default=TRAIN_DIR)
        sp.add_argument("--val_dir", default=VAL_DIR)
        sp.add_argument("--model_dir", default=MODEL_DIR)
        sp.add_argument("--batch_size", type=int, default=BATCH)
        sp.add_argument("--seed", type=int, default=SEED)

        # search domain (match code)
        sp.add_argument("--head_type", nargs="+", default=["baseline", "gap", "gap_gmp", "swish", "mlp2"])
        sp.add_argument("--dense_units", nargs="+", type=int, default=[256, 384, 512])
        sp.add_argument("--drop1", nargs="+", type=float, default=[0.2, 0.3, 0.4])
        sp.add_argument("--drop2", nargs="+", type=float, default=[0.2, 0.3, 0.4])
        sp.add_argument("--dense_l2", nargs="+", type=float, default=[0.0, 1e-5, 1e-4])
        sp.add_argument("--dense_units2", nargs="+", type=int, default=[128, 192, 256])
        sp.add_argument("--drop3", nargs="+", type=float, default=[0.2, 0.3])
        sp.add_argument("--lr1", nargs="+", type=float, default=[1e-3, 5e-4, 3e-4])
        sp.add_argument("--lr2", nargs="+", type=float, default=[1e-4, 5e-5, 3e-5])
        sp.add_argument("--unfreeze", nargs="+", type=int, default=[20, 40, 60])

        sp.add_argument("--search_method", choices=["random", "grid"], default="random")
        sp.add_argument("--trials", type=int, default=30)
        sp.add_argument("--proxy_phase1_epochs", type=int, default=2)
        sp.add_argument("--proxy_phase2_epochs", type=int, default=4)
        sp.add_argument("--proxy_patience", type=int, default=2)

    t = sub.add_parser("tune")
    add_common(t)

    tr = sub.add_parser("train")
    add_common(tr)
    tr.add_argument("--require_tuned", type=int, default=1)
    tr.add_argument("--epochs_phase1", type=int, default=E1)
    tr.add_argument("--epochs_phase2", type=int, default=E2)

    ev = sub.add_parser("eval")
    ev.add_argument("--val_dir", default=VAL_DIR)
    ev.add_argument("--model_dir", default=MODEL_DIR)
    ev.add_argument("--batch_size", type=int, default=BATCH)
    ev.add_argument("--seed", type=int, default=SEED)
    ev.add_argument("--weights", default=os.path.join(MODEL_DIR, "best_classification_model.weights.h5"))
    ev.add_argument("--out_prefix", default=os.path.join(MODEL_DIR, "eval"))
    ev.add_argument("--grid_n", type=int, default=16)
    ev.add_argument("--grid_per_class", type=int, default=None)

    return p.parse_args()

def main():
    args = parse_args()
    setup_runtime(args.seed)

    if args.cmd == "tune":
        run_tune(args)
    elif args.cmd == "train":
        run_train(args)
    elif args.cmd == "eval":
        run_eval(args)
    else:
        raise ValueError(args.cmd)

if __name__ == "__main__":
    main()