# tools/segmentation_preprocessing.py
import argparse
import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import shutil

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

GLOBAL_SEED = 42

Q1_DEFAULT = 0.33
Q2_DEFAULT = 0.66

CLS_ALS = "ALTERNARIA LEAF SPOT"
CLS_LS = "LEAF SPOT (EARLY AND LATE)"
CLS_ROS = "ROSETTE"
CLS_RUST = "RUST"

DEFAULT_CLASS_WEIGHT_OVERRIDES = {
    CLS_ALS: {"small": 1.6, "mid": 1.0, "large": 0.8},
    CLS_LS: {"small": 1.5, "mid": 1.0, "large": 0.9},
    CLS_ROS: {"small": 1.0, "mid": 1.0, "large": 1.0},
    CLS_RUST: {"small": 1.4, "mid": 1.0, "large": 0.9},
}

# =========================
# Helpers (filesystem)
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_files(path: str, exts: Tuple[str, ...]) -> List[str]:
    if not os.path.exists(path):
        return []
    out = []
    for fn in sorted(os.listdir(path)):
        fp = os.path.join(path, fn)
        if os.path.isfile(fp) and fn.lower().endswith(tuple(e.lower() for e in exts)):
            out.append(fn)
    return out


def list_classes(split_root: str) -> List[str]:
    if not os.path.exists(split_root):
        return []
    classes = []
    for d in sorted(os.listdir(split_root)):
        full = os.path.join(split_root, d)
        if os.path.isdir(full) and not d.startswith(".") and not d.startswith("_"):
            classes.append(d)
    return classes


def find_mask(mask_dir: str, stem: str) -> Optional[str]:
    for ext in MASK_EXT:
        p = os.path.join(mask_dir, f"{stem}{ext}")
        if os.path.exists(p):
            return p
    return None


# =========================
# Orphans checks
# =========================

def _count_orphans(orphans_dir: str) -> Dict[str, int]:
    no_mask_dir = os.path.join(orphans_dir, "no_mask")
    no_image_dir = os.path.join(orphans_dir, "no_image")

    def _count_files(root: str) -> int:
        if not os.path.exists(root):
            return 0
        total = 0
        for r, _, files in os.walk(root):
            for f in files:
                if not f.startswith("."):
                    total += 1
        return total

    c_no_mask = _count_files(no_mask_dir)
    c_no_image = _count_files(no_image_dir)
    return {
        "no_mask": c_no_mask,
        "no_image": c_no_image,
        "total": c_no_mask + c_no_image,
    }


# =========================
# Lesion statistics
# =========================

def mask_area_ratio(mask_path: str) -> float:
    m = Image.open(mask_path).convert("L")
    arr = np.asarray(m, dtype=np.uint8)
    binm = (arr >= 128).astype(np.float32)
    return float(binm.mean())

def compute_quantile_cutoffs(mask_paths: List[str], q1: float, q2: float) -> Dict[str, float]:
    ratios = []
    for mp in mask_paths:
        try:
            ratios.append(mask_area_ratio(mp))
        except Exception:
            ratios.append(0.0)

    if not ratios:
        return {"q1": 0.0, "q2": 0.0}

    rnp = np.asarray(ratios, dtype=np.float32)
    q1v = float(np.quantile(rnp, float(q1)))
    q2v = float(np.quantile(rnp, float(q2)))
    return {"q1": q1v, "q2": q2v}

def bucket_by_quantile(r: float, q1v: float, q2v: float) -> str:
    if r < q1v:
        return "small"
    if r < q2v:
        return "mid"
    return "large"

# =========================
# Weight profiles (STRICT)
# =========================

def load_weight_overrides(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    if not path:
        return dict(DEFAULT_CLASS_WEIGHT_OVERRIDES)

    import json
    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)

    merged = dict(DEFAULT_CLASS_WEIGHT_OVERRIDES)
    for cls, prof in user.items():
        merged[str(cls)] = {
            "small": float(prof.get("small", 1.0)),
            "mid": float(prof.get("mid", 1.0)),
            "large": float(prof.get("large", 1.0)),
        }
    return merged


def assert_strict_weights(train_classes: List[str], weight_overrides: Dict[str, Dict[str, float]]) -> None:
    missing = [c for c in train_classes if c not in weight_overrides]
    if missing:
        raise RuntimeError(
            "STRICT weights violation: class ditemukan di TRAIN tapi tidak ada di weight map.\n"
            f"Missing classes: {missing}\n"
            "Tambahkan ke DEFAULT_CLASS_WEIGHT_OVERRIDES atau berikan --weight_override_json."
        )


# =========================
# Build datasets
# =========================

def _copy_pair(img_src: str, mask_src: str, img_dst: str, mask_dst: str) -> None:
    ensure_dir(os.path.dirname(img_dst))
    ensure_dir(os.path.dirname(mask_dst))
    shutil.copy2(img_src, img_dst)
    shutil.copy2(mask_src, mask_dst)


def _augment_and_save_pair(img_src: str, mask_src: str, img_dst: str, mask_dst: str, rng: np.random.RandomState) -> None:
    ensure_dir(os.path.dirname(img_dst))
    ensure_dir(os.path.dirname(mask_dst))

    img = Image.open(img_src).convert("RGB")
    msk = Image.open(mask_src).convert("L")

    out_w, out_h = img.size

    x = np.asarray(img, np.float32) / 255.0
    y = np.asarray(msk, np.uint8)
    y = (y >= 128).astype(np.uint8)

    # flip Horizontal
    if rng.rand() < 0.5:
        x = x[:, ::-1, :]
        y = y[:, ::-1]

    # flip Vertikal
    if rng.rand() < 0.2:
        x = x[::-1, :, :]
        y = y[::-1, :]

    # rotate 0/90/180/270
    k = int(rng.randint(0, 4))
    if k:
        x = np.rot90(x, k, axes=(0, 1)).copy()
        y = np.rot90(y, k, axes=(0, 1)).copy()

        if x.shape[0] != out_h or x.shape[1] != out_w:
            x_img = Image.fromarray(np.clip(x * 255.0, 0, 255).astype(np.uint8), mode="RGB")
            x_img = x_img.resize((out_w, out_h), Image.BILINEAR)
            x = np.asarray(x_img, np.float32) / 255.0

            y_img = Image.fromarray((y > 0).astype(np.uint8) * 255, mode="L")
            y_img = y_img.resize((out_w, out_h), Image.NEAREST)
            y = (np.asarray(y_img, np.uint8) >= 128).astype(np.uint8)

    # photometric (image only)
    if rng.rand() < 0.35:
        c = float(rng.uniform(0.90, 1.10))
        b = float(rng.uniform(-0.06, 0.06))
        x = np.clip(x * c + b, 0.0, 1.0)

    # save
    out_img = Image.fromarray(np.clip(x * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    out_msk = Image.fromarray((y > 0).astype(np.uint8) * 255, mode="L")

    out_img.save(img_dst)
    out_msk.save(mask_dst)

def _sample_indices_weighted(buckets: Dict[str, List[int]], weights: Dict[str, float], k: int) -> List[int]:
    population = []
    probs = []
    for b, idxs in buckets.items():
        if not idxs:
            continue
        w = float(weights.get(b, 1.0))
        for ix in idxs:
            population.append(ix)
            probs.append(w)

    if not population:
        return []

    probs_np = np.asarray(probs, dtype=np.float64)
    probs_np = probs_np / probs_np.sum()
    chosen = np.random.choice(len(population), size=k, replace=True, p=probs_np)
    return [population[i] for i in chosen.tolist()]

def build_train_balanced_global(src_train_root: str, dst_root: str, cutoffs: Dict[str, float],
    profiles: Dict[str, Dict[str, float]], global_r_max: float, overwrite: bool) -> None:
    if os.path.exists(dst_root) and (not overwrite):
        print(f"[SKIP] dst already exists (no_overwrite): {dst_root}")
        return

    if os.path.exists(dst_root) and overwrite:
        import shutil
        shutil.rmtree(dst_root)
    ensure_dir(dst_root)

    rng = np.random.RandomState(GLOBAL_SEED)

    q1v = float(cutoffs["q1"])
    q2v = float(cutoffs["q2"])

    classes = list_classes(src_train_root)
    assert_strict_weights(classes, profiles)

    class_pairs: Dict[str, List[Tuple[str, str, float, str]]] = {}
    counts = []

    for cls in classes:
        img_dir = os.path.join(src_train_root, cls, "images")
        mask_dir = os.path.join(src_train_root, cls, "masks")
        imgs = list_files(img_dir, IMG_EXT)

        pairs = []
        for fn in imgs:
            stem = os.path.splitext(fn)[0]
            mp = find_mask(mask_dir, stem)
            if not mp:
                continue
            ip = os.path.join(img_dir, fn)
            try:
                r = mask_area_ratio(mp)
            except Exception:
                r = 0.0
            b = bucket_by_quantile(r, q1v, q2v)
            pairs.append((ip, mp, r, b))

        if pairs:
            class_pairs[cls] = pairs
            counts.append(len(pairs))

    if not class_pairs:
        raise RuntimeError("Tidak ada pasangan image-mask valid di TRAIN untuk global build.")

    n_min = int(min(counts))
    n_max = int(max(counts))

    if float(global_r_max) > 0.0:
        cap = int(max(1, float(global_r_max) * float(n_min)))
        target = min(n_max, cap)
    else:
        target = n_max

    for cls, pairs in class_pairs.items():
        dst_img_dir = os.path.join(dst_root, cls, "images")
        dst_mask_dir = os.path.join(dst_root, cls, "masks")
        ensure_dir(dst_img_dir)
        ensure_dir(dst_mask_dir)

        buckets = {"small": [], "mid": [], "large": []}
        for i, (_, __, ___, b) in enumerate(pairs):
            buckets[b].append(i)

        for ip, mp, _, __ in pairs:
            _copy_pair(ip, mp, os.path.join(dst_img_dir, os.path.basename(ip)), os.path.join(dst_mask_dir, os.path.basename(mp)))

        need = max(0, target - len(pairs))
        if need <= 0:
            continue

        sampled = _sample_indices_weighted(buckets, profiles[cls], need)
        for j, idx in enumerate(sampled):
            ip, mp, _, __ = pairs[idx]
            base_img = os.path.splitext(os.path.basename(ip))[0]
            base_mask = os.path.splitext(os.path.basename(mp))[0]
            img_dst = os.path.join(dst_img_dir, f"aug_{j:05d}_{base_img}{os.path.splitext(ip)[1]}")
            mask_dst = os.path.join(dst_mask_dir, f"aug_{j:05d}_{base_mask}{os.path.splitext(mp)[1]}")
            _augment_and_save_pair(ip, mp, img_dst, mask_dst, rng)

    print("[OK] train_balanced_global built:", dst_root)


# =========================
# Pipeline runner
# =========================

def run_pipeline(args: argparse.Namespace) -> None:
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    if hasattr(args, "fail_on_orphans") and int(getattr(args, "fail_on_orphans", 1)) == 1:
        counts = _count_orphans(getattr(args, "orphans_dir", "datasets/results/orphans_report"))
        if counts["total"] > 0:
            raise RuntimeError(
                "Orphans terdeteksi pada dataset split (image tanpa mask / mask tanpa image). "
                f"no_mask={counts['no_mask']} no_image={counts['no_image']} "
                f"di '{getattr(args, 'orphans_dir', 'datasets/results/orphans_report')}'. "
                "Perbaiki dataset (hapus/repair pasangan) atau jalankan split ulang sebelum preprocessing."
            )

    src_train = os.path.join(args.src_base, args.train_split)
    src_val = os.path.join(args.src_base, args.val_split)

    dst_global = os.path.join(args.dest_base, "train_balanced_global")

    train_masks = []
    for cls in list_classes(src_train):
        mask_dir = os.path.join(src_train, cls, "masks")
        for fn in list_files(mask_dir, MASK_EXT):
            train_masks.append(os.path.join(mask_dir, fn))

    cutoffs = compute_quantile_cutoffs(train_masks, args.q1, args.q2)
    print(f"[INFO] quantile cutoffs (train masks): q1={cutoffs['q1']:.6f} q2={cutoffs['q2']:.6f}")

    weight_overrides = load_weight_overrides(args.weight_override_json)

    overwrite = not bool(args.no_overwrite)

    build_train_balanced_global(
        src_train_root=src_train,
        dst_root=dst_global,
        cutoffs=cutoffs,
        profiles=weight_overrides,
        global_r_max=float(args.global_r_max),
        overwrite=overwrite,
    )
    print("\n[OK] Global balanced dataset ready:")
    print("  -", dst_global)

    _ = src_val


# =========================
# CLI
# =========================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Segmentation preprocessing: build global balanced train set (STRICT weights). No JSON outputs."
    )

    ap.add_argument("--src_base", default="datasets/processed/segmentation_dataset")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="val")
    ap.add_argument("--dest_base", default="datasets/processed/segmentation_dataset")

    ap.add_argument(
        "--global_r_max",
        type=float,
        default=0.0,
        help="Cap factor vs n_min (0 disables cap => balance to max_n).",
    )

    ap.add_argument("--q1", type=float, default=Q1_DEFAULT)
    ap.add_argument("--q2", type=float, default=Q2_DEFAULT)

    ap.add_argument("--weight_override_json", default=None)

    ap.add_argument("--no_overwrite", action="store_true")

    ap.add_argument(
        "--orphans_dir",
        default="datasets/results/orphans_report",
        help="Folder orphans_report dari dataset_split.py (no_mask/no_image).",
    )
    ap.add_argument(
        "--fail_on_orphans",
        type=int,
        default=1,
        help="1=raise error jika ada orphans terdeteksi (default 1).",
    )

    args = ap.parse_args()
    run_pipeline(args)
