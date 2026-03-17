# tools/severity_preprocessing.py
import argparse
import os
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(d: Path):
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.suffix.lower() in VALID_EXT])

def k(sz: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))


# ============================================================
# HAND / SKIN VETO
# ============================================================
def overlap_ratio(a01: np.ndarray, b01: np.ndarray) -> float:
    denom = float(a01.sum())
    if denom <= 0:
        return 0.0
    return float((a01 & b01).sum()) / denom


# ============================================================
# LEAF-LIKENESS FEATURES
# ============================================================

def exg_map(bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(bgr.astype(np.float32))
    exg = 2.0 * g - r - b
    exg = (exg - exg.min()) / (exg.max() - exg.min() + 1e-6)
    return exg


def sat_map(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1].astype(np.float32) / 255.0


def lap_var(gray: np.ndarray, mask01: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    vals = lap[mask01 > 0]
    if vals.size < 50:
        return 1e9
    return float(vals.var())


def solidity(mask01: np.ndarray) -> float:
    cnts, _ = cv2.findContours(mask01.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area <= 1:
        return 0.0
    hull = cv2.convexHull(cnt)
    ha = float(cv2.contourArea(hull))
    if ha <= 1:
        return 0.0
    return area / ha


def circularity(mask01: np.ndarray) -> float:
    cnts, _ = cv2.findContours(mask01.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    if area <= 1 or peri <= 1:
        return 0.0
    return (4.0 * np.pi * area) / (peri * peri)


def leaf_score(bgr: np.ndarray, mask01: np.ndarray, args, skin01: np.ndarray) -> float:
    if mask01.sum() == 0:
        return -1e9

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    exg = exg_map(bgr)
    sat = sat_map(bgr)

    exg_mean = float(exg[mask01 > 0].mean())
    sat_mean = float(sat[mask01 > 0].mean())
    tex = -np.log1p(lap_var(gray, mask01))
    sol = solidity(mask01)
    circ = circularity(mask01)
    skin_ov = overlap_ratio(mask01, skin01)

    return float(
        args.w_exg * exg_mean +
        args.w_sat * sat_mean +
        args.w_tex * tex +
        args.w_sol * sol +
        args.w_circ * circ -
        args.w_skin * skin_ov
    )


# ============================================================
# SAM BUILD
# ============================================================

def build_amg(args):
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    sam = sam_model_registry[args.model](checkpoint=args.sam_ckpt)
    sam.to(device=device)

    amg = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_overlap_ratio=args.crop_overlap_ratio,
        min_mask_region_area=args.min_mask_region_area,
        output_mode="binary_mask",
    )
    return amg, device

# ============================================================
# AUGMENTATION (TRAIN ONLY)
# ============================================================

def augment_train_split(root: Path):
    train_img = root / "train" / "images"
    train_mask = root / "train" / "masks"

    if not train_img.exists() or not train_mask.exists():
        raise SystemExit("train/images or train/masks not found")

    imgs = list_images(train_img)
    print(f"[AUGMENT] Found {len(imgs)} training samples")

    added = 0
    for ip in tqdm(imgs, desc="Augmenting train") if tqdm else imgs:
        mp = train_mask / f"{ip.stem}.png"
        if not mp.exists():
            continue

        img = cv2.imread(str(ip))
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        cv2.imwrite(str(train_img / f"{ip.stem}_hflip{ip.suffix}"), cv2.flip(img, 1))
        cv2.imwrite(str(train_mask / f"{ip.stem}_hflip.png"), cv2.flip(mask, 1))

        cv2.imwrite(str(train_img / f"{ip.stem}_vflip{ip.suffix}"), cv2.flip(img, 0))
        cv2.imwrite(str(train_mask / f"{ip.stem}_vflip.png"), cv2.flip(mask, 0))

        added += 2

    print(f"[AUGMENT] Added {added} samples")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/processed/severity_dataset")
    ap.add_argument("--augment_train", action="store_true")

    # SAM args (only needed if NOT augmenting)
    ap.add_argument("--sam_ckpt", default=None)
    ap.add_argument("--model", default="vit_b")
    ap.add_argument("--device", default="auto")

    # parameters (kept for reproducibility)
    ap.add_argument("--points_per_side", type=int, default=40)
    ap.add_argument("--pred_iou_thresh", type=float, default=0.88)
    ap.add_argument("--stability_score_thresh", type=float, default=0.90)
    ap.add_argument("--box_nms_thresh", type=float, default=0.70)
    ap.add_argument("--crop_n_layers", type=int, default=2)
    ap.add_argument("--crop_overlap_ratio", type=float, default=0.35)
    ap.add_argument("--min_mask_region_area", type=int, default=900)

    ap.add_argument("--w_exg", type=float, default=1.2)
    ap.add_argument("--w_sat", type=float, default=0.9)
    ap.add_argument("--w_tex", type=float, default=0.35)
    ap.add_argument("--w_sol", type=float, default=1.0)
    ap.add_argument("--w_circ", type=float, default=0.2)
    ap.add_argument("--w_skin", type=float, default=4.0)

    args = ap.parse_args()
    root = Path(args.root)

    # MODE B: AUGMENT ONLY
    if args.augment_train:
        augment_train_split(root)
        print("Augmentation finished.")
        return

    # MODE A: SAM REQUIRED
    if args.sam_ckpt is None:
        raise SystemExit("--sam_ckpt is required unless --augment_train is used")

    print("Leaf GT generation with SAM is configured. (Pipeline unchanged)")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
