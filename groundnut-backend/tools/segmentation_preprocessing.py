# tools/segmentation_preprocessing.py
#
# STRICT behavior:
#   ✅ Every class found in TRAIN must have weights defined in DEFAULT_CLASS_WEIGHT_OVERRIDES
#      (or overridden via --weight_override_json).
#   ✅ No fallback weights -> if an unexpected class appears, script stops (safer & reproducible).
#
# Builds:
#   1) train_balanced_perclass  -> for per-class (binary) segmentation models
#   2) train_balanced_global    -> for 1 global (multi-class) segmentation model
#
# Dataset structure expected:
#   <src_base>/<split>/<CLASS>/images/*
#   <src_base>/<split>/<CLASS>/masks/*
#
# Default working directory: groundnut-backend/
# Default paths:
#   src_base:  datasets/processed/segmentation_dataset
#   orphans:   datasets/results/orphans_report
#
import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image


# =========================
# Constants / Config
# =========================

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

GLOBAL_SEED = 42

# Quantile cutoffs default
Q1_DEFAULT = 0.33
Q2_DEFAULT = 0.66

# Per-class target cap control (ke n_min)
R_MAX_DEFAULT = 5.0

# Common class names (sesuaikan dengan dataset kamu)
CLS_ALS = "ALTERNARIA LEAF SPOT"
CLS_LS = "LEAF SPOT (EARLY AND LATE)"
CLS_ROS = "ROSETTE"
CLS_RUST = "RUST"
CLS_HEALTHY = "HEALTHY"

# STRICT: semua class di TRAIN harus ada di map ini (kecuali di override JSON)
# Angka di sini adalah "prioritas augment" per bucket lesi (small/mid/large),
# dan weight ini dipakai untuk sampling mask berdasarkan lesion size.
DEFAULT_CLASS_WEIGHT_OVERRIDES = {
    CLS_ALS: {"small": 1.6, "mid": 1.0, "large": 0.8},
    CLS_LS: {"small": 1.5, "mid": 1.0, "large": 0.9},
    CLS_ROS: {"small": 1.0, "mid": 1.0, "large": 1.0},  # rosette default no priority
    CLS_RUST: {"small": 1.4, "mid": 1.0, "large": 0.9},
    # HEALTHY biasanya tidak disertakan untuk segmentation; kalau dataset kamu menyertakan, definisikan.
    # CLS_HEALTHY: {"small": 1.0, "mid": 1.0, "large": 1.0},
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
    """
    orphans_dir expected structure:
      orphans_report/
        no_mask/<CLASS>/*
        no_image/<CLASS>/*
    """
    no_mask_dir = os.path.join(orphans_dir, "no_mask")
    no_image_dir = os.path.join(orphans_dir, "no_image")

    def _count_files(root: str) -> int:
        if not os.path.exists(root):
            return 0
        total = 0
        for r, _, files in os.walk(root):
            for f in files:
                # count everything (images/masks)
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
    """Return lesion area ratio (0..1)."""
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
    """
    Optional JSON input to override DEFAULT_CLASS_WEIGHT_OVERRIDES.
    JSON format:
      {
        "RUST": {"small": 1.4, "mid": 1.0, "large": 0.9},
        ...
      }
    """
    if not path:
        return dict(DEFAULT_CLASS_WEIGHT_OVERRIDES)

    import json
    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)

    # merge (user takes precedence)
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
    # copy with PIL save to normalize extensions? (keep original bytes safer: use shutil)
    import shutil
    shutil.copy2(img_src, img_dst)
    shutil.copy2(mask_src, mask_dst)


def _sample_indices_weighted(buckets: Dict[str, List[int]], weights: Dict[str, float], k: int) -> List[int]:
    """
    Weighted sampling over bucketed indices; sampling with replacement.
    """
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


def build_train_balanced_perclass(
    src_train_root: str,
    dst_root: str,
    cutoffs: Dict[str, float],
    weight_overrides: Dict[str, Dict[str, float]],
    r_max: float,
    preset_targets: Optional[Dict[str, int]],
    overwrite: bool,
) -> None:
    """
    For each class, create balanced binary train dataset under:
      dst_root/<CLASS>/images
      dst_root/<CLASS>/masks
    """
    if os.path.exists(dst_root) and (not overwrite):
        print(f"[SKIP] dst already exists (no_overwrite): {dst_root}")
        return

    # Reset folder
    if os.path.exists(dst_root) and overwrite:
        import shutil
        shutil.rmtree(dst_root)
    ensure_dir(dst_root)

    q1v = float(cutoffs["q1"])
    q2v = float(cutoffs["q2"])

    classes = list_classes(src_train_root)
    assert_strict_weights(classes, weight_overrides)

    # gather per-class pairs + lesion ratio buckets
    class_pairs: Dict[str, List[Tuple[str, str, float, str]]] = {}

    n_min = None
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

        if not pairs:
            print(f"[WARN] no pairs for class: {cls} (skipped)")
            continue

        class_pairs[cls] = pairs
        if n_min is None:
            n_min = len(pairs)
        else:
            n_min = min(n_min, len(pairs))

    if not class_pairs:
        raise RuntimeError("Tidak ada pasangan image-mask valid di TRAIN.")

    n_min = int(n_min or 0)
    cap = int(max(1, float(r_max) * float(n_min)))

    # choose targets per class
    targets: Dict[str, int] = {}
    for cls, pairs in class_pairs.items():
        if preset_targets and cls in preset_targets:
            targets[cls] = int(preset_targets[cls])
        else:
            # default: cap growth to r_max*n_min, but never shrink originals
            targets[cls] = min(cap, max(len(pairs), n_min))

    print("\n=== Per-class targets (no JSON output) ===")
    for cls in sorted(targets.keys()):
        print(f"  - {cls:<27}: {len(class_pairs[cls])} -> {targets[cls]}")
    print("=========================================\n")

    # build each class
    for cls, pairs in class_pairs.items():
        dst_img_dir = os.path.join(dst_root, cls, "images")
        dst_mask_dir = os.path.join(dst_root, cls, "masks")
        ensure_dir(dst_img_dir)
        ensure_dir(dst_mask_dir)

        # bucket indices
        buckets = {"small": [], "mid": [], "large": []}
        for i, (_, __, ___, b) in enumerate(pairs):
            buckets[b].append(i)

        weights = weight_overrides[cls]
        target = targets[cls]

        # always include originals (copy all unique originals once)
        for i, (ip, mp, _, __) in enumerate(pairs):
            _copy_pair(ip, mp, os.path.join(dst_img_dir, os.path.basename(ip)), os.path.join(dst_mask_dir, os.path.basename(mp)))

        # sample extra to reach target
        need = max(0, target - len(pairs))
        if need <= 0:
            continue

        sampled = _sample_indices_weighted(buckets, weights, need)
        for j, idx in enumerate(sampled):
            ip, mp, _, __ = pairs[idx]
            base_img = os.path.splitext(os.path.basename(ip))[0]
            base_mask = os.path.splitext(os.path.basename(mp))[0]
            img_dst = os.path.join(dst_img_dir, f"aug_{j:05d}_{base_img}{os.path.splitext(ip)[1]}")
            mask_dst = os.path.join(dst_mask_dir, f"aug_{j:05d}_{base_mask}{os.path.splitext(mp)[1]}")
            _copy_pair(ip, mp, img_dst, mask_dst)

    print("[OK] train_balanced_perclass built:", dst_root)


def build_train_balanced_global(
    src_train_root: str,
    dst_root: str,
    cutoffs: Dict[str, float],
    profiles: Dict[str, Dict[str, float]],
    global_r_max: float,
    overwrite: bool,
) -> None:
    """
    Build a single global-balanced train dataset:
      dst_root/<CLASS>/images
      dst_root/<CLASS>/masks
    Strategy: balance each class up to max_n (or capped by global_r_max * n_min if global_r_max > 0).
    """
    if os.path.exists(dst_root) and (not overwrite):
        print(f"[SKIP] dst already exists (no_overwrite): {dst_root}")
        return

    # reset
    if os.path.exists(dst_root) and overwrite:
        import shutil
        shutil.rmtree(dst_root)
    ensure_dir(dst_root)

    q1v = float(cutoffs["q1"])
    q2v = float(cutoffs["q2"])

    classes = list_classes(src_train_root)
    assert_strict_weights(classes, profiles)

    # gather per-class pairs
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

    # target balancing
    if float(global_r_max) > 0.0:
        cap = int(max(1, float(global_r_max) * float(n_min)))
        target = min(n_max, cap)
    else:
        target = n_max

    print("\n=== Global balanced target (no JSON output) ===")
    print(f"n_min={n_min}  n_max={n_max}  global_target={target}")
    print("=============================================\n")

    # build
    for cls, pairs in class_pairs.items():
        dst_img_dir = os.path.join(dst_root, cls, "images")
        dst_mask_dir = os.path.join(dst_root, cls, "masks")
        ensure_dir(dst_img_dir)
        ensure_dir(dst_mask_dir)

        buckets = {"small": [], "mid": [], "large": []}
        for i, (_, __, ___, b) in enumerate(pairs):
            buckets[b].append(i)

        # copy originals
        for ip, mp, _, __ in pairs:
            _copy_pair(ip, mp, os.path.join(dst_img_dir, os.path.basename(ip)), os.path.join(dst_mask_dir, os.path.basename(mp)))

        # augment sample to target
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
            _copy_pair(ip, mp, img_dst, mask_dst)

    print("[OK] train_balanced_global built:", dst_root)


# =========================
# Pipeline runner
# =========================

def run_pipeline(args: argparse.Namespace) -> None:
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    # Fail fast jika ada orphans (dataset tidak konsisten)
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

    # Build outputs
    dst_perclass = os.path.join(args.dest_base, "train_balanced_perclass")
    dst_global = os.path.join(args.dest_base, "train_balanced_global")

    # Collect all TRAIN masks for quantiles
    train_masks = []
    for cls in list_classes(src_train):
        mask_dir = os.path.join(src_train, cls, "masks")
        for fn in list_files(mask_dir, MASK_EXT):
            train_masks.append(os.path.join(mask_dir, fn))

    cutoffs = compute_quantile_cutoffs(train_masks, args.q1, args.q2)
    print(f"[INFO] quantile cutoffs (train masks): q1={cutoffs['q1']:.6f} q2={cutoffs['q2']:.6f}")

    # Strict weight profiles
    weight_overrides = load_weight_overrides(args.weight_override_json)

    # preset targets input (optional) - input only, no output json
    preset_targets = None
    if args.preset_targets_json:
        import json
        with open(args.preset_targets_json, "r", encoding="utf-8") as f:
            preset_targets = json.load(f)

    overwrite = not bool(args.no_overwrite)

    if args.build_mode in ("perclass", "both"):
        build_train_balanced_perclass(
            src_train_root=src_train,
            dst_root=dst_perclass,
            cutoffs=cutoffs,
            weight_overrides=weight_overrides,
            r_max=float(args.r_max),
            preset_targets=preset_targets,
            overwrite=overwrite,
        )
        print("\n[OK] Per-class dataset ready:")
        print("  -", dst_perclass)

    if args.build_mode in ("global", "both"):
        build_train_balanced_global(
            src_train_root=src_train,
            dst_root=dst_global,
            cutoffs=cutoffs,
            profiles=weight_overrides,
            global_r_max=float(args.global_r_max),
            overwrite=overwrite,
        )
        print("\n[OK] Global dataset ready:")
        print("  -", dst_global)

    # Note: we do not write any stats/json files anymore.
    # src_val is currently unused for writing outputs; kept for future validation if needed.
    _ = src_val


# =========================
# CLI
# =========================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Segmentation preprocessing: build perclass/global balanced train sets (STRICT weights). No JSON outputs."
    )

    ap.add_argument("--src_base", default="datasets/processed/segmentation_dataset")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="val")
    ap.add_argument("--dest_base", default="datasets/processed/segmentation_dataset")

    ap.add_argument("--build_mode", choices=["perclass", "global", "both"], default="perclass")

    ap.add_argument("--r_max", type=float, default=R_MAX_DEFAULT)
    ap.add_argument(
        "--global_r_max",
        type=float,
        default=0.0,
        help="Cap factor vs n_min (0 disables cap => balance to max_n).",
    )

    ap.add_argument("--q1", type=float, default=Q1_DEFAULT)
    ap.add_argument("--q2", type=float, default=Q2_DEFAULT)

    ap.add_argument("--preset_targets_json", default=None)
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
