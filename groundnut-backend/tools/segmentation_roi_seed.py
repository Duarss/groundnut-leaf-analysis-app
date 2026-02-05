# tools/segmentation_roi_seed.py
import os, glob, shutil, argparse, random
from pathlib import Path
import numpy as np
from PIL import Image

IMG_EXT  = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SEED = 42

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def is_image_file(p):
    return p.lower().endswith(IMG_EXT)

def find_mask(mask_dir, stem):
    for ext in MASK_EXT:
        mp = os.path.join(mask_dir, stem + ext)
        if os.path.exists(mp):
            return mp
    return None

def collect_pairs(split_root):
    out = {}
    if not os.path.isdir(split_root):
        return out
    for cls in sorted(os.listdir(split_root)):
        img_dir = os.path.join(split_root, cls, "images")
        msk_dir = os.path.join(split_root, cls, "masks")
        if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
            continue
        pairs = []
        for ip in sorted(glob.glob(os.path.join(img_dir, "*"))):
            if not is_image_file(ip):
                continue
            stem = Path(ip).stem
            mp = find_mask(msk_dir, stem)
            if mp:
                pairs.append((ip, mp))
        if pairs:
            out[cls] = pairs
    return out

# ---------------- ROI helpers ----------------
def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def crop_with_center(img_arr, mask_arr, cx, cy, roi_w, roi_h):
    H, W = mask_arr.shape[:2]
    x1 = int(cx - roi_w // 2)
    y1 = int(cy - roi_h // 2)
    x1 = max(0, min(x1, W - roi_w))
    y1 = max(0, min(y1, H - roi_h))
    x2 = x1 + roi_w
    y2 = y1 + roi_h
    img_crop = img_arr[y1:y2, x1:x2, :]
    msk_crop = mask_arr[y1:y2, x1:x2]
    return img_crop, msk_crop, (x1, y1, x2, y2)

def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-9)

def connected_components(mask01):
    """Tiny 4-neighborhood CC without scipy. Returns list of comps, each comp list[(y,x)]."""
    H, W = mask01.shape
    visited = np.zeros((H, W), dtype=np.uint8)
    comps = []
    for y in range(H):
        for x in range(W):
            if mask01[y, x] == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = 1
            comp = []
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                if cy > 0 and mask01[cy-1, cx] and not visited[cy-1, cx]:
                    visited[cy-1, cx] = 1; stack.append((cy-1, cx))
                if cy+1 < H and mask01[cy+1, cx] and not visited[cy+1, cx]:
                    visited[cy+1, cx] = 1; stack.append((cy+1, cx))
                if cx > 0 and mask01[cy, cx-1] and not visited[cy, cx-1]:
                    visited[cy, cx-1] = 1; stack.append((cy, cx-1))
                if cx+1 < W and mask01[cy, cx+1] and not visited[cy, cx+1]:
                    visited[cy, cx+1] = 1; stack.append((cy, cx+1))
            comps.append(comp)
    return comps

def pick_seed_component(mask01, rng, min_comp_pixels=20):
    """Pick random CC weighted by size; return centroid (x,y)."""
    comps = connected_components(mask01)
    comps = [c for c in comps if len(c) >= min_comp_pixels]
    if not comps:
        return None
    sizes = np.array([len(c) for c in comps], dtype=np.float64)
    probs = sizes / (sizes.sum() + 1e-9)
    idx = int(rng.choice(len(comps), p=probs))
    comp = comps[idx]
    ys = np.array([p[0] for p in comp], dtype=np.float32)
    xs = np.array([p[1] for p in comp], dtype=np.float32)
    cy = int(np.round(ys.mean()))
    cx = int(np.round(xs.mean()))
    return cx, cy

def resize_to_canvas(img_pil, msk_pil, canvas_h, canvas_w):
    """ROI-from-resized: resize FIRST to canvas (W,H), then crop in that space."""
    img_rs = img_pil.resize((canvas_w, canvas_h), Image.BILINEAR)
    msk_rs = msk_pil.resize((canvas_w, canvas_h), Image.NEAREST)
    img_arr = np.asarray(img_rs, np.uint8)
    msk_arr = (np.asarray(msk_rs, np.uint8) > 127).astype(np.uint8)
    return img_arr, msk_arr

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_base", default="datasets/processed/segmentation_dataset")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--dest_train_split", default="train_roi")

    # canvas must match how you resize during training/val (exe_segmentation_model.py)
    ap.add_argument("--canvas_h", type=int, default=480)
    ap.add_argument("--canvas_w", type=int, default=640)

    # ROI output (square, for bbox workflow)
    ap.add_argument("--roi", type=int, default=384, help="Final ROI size (square). Default 384.")

    ap.add_argument("--n_rois_per_image", type=int, default=2,
                    help="How many ROI crops per image (train only). Recommended 1-2.")
    ap.add_argument("--jitter", type=float, default=0.40,
                    help="Jitter center ±fraction ROI size (on canvas space).")
    ap.add_argument("--min_mask_ratio", type=float, default=0.001,
                    help="Minimum lesion ratio inside ROI on canvas space. Gentle default.")
    ap.add_argument("--max_tries", type=int, default=20,
                    help="Max attempts per ROI to satisfy constraints.")
    ap.add_argument("--max_iou", type=float, default=0.35,
                    help="Max allowed IoU overlap between ROIs from the same image (canvas coords).")
    ap.add_argument("--scale_min", type=float, default=0.90,
                    help="Random ROI scale min (relative to roi).")
    ap.add_argument("--scale_max", type=float, default=1.10,
                    help="Random ROI scale max (relative to roi).")
    ap.add_argument("--min_comp_pixels", type=int, default=20,
                    help="Min CC pixels (on canvas) to be used as seed component.")
    ap.add_argument("--only_class", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rng = np.random.RandomState(SEED)
    random.seed(SEED)

    canvas_h, canvas_w = int(args.canvas_h), int(args.canvas_w)
    roi = int(args.roi)
    roi_h, roi_w = roi, roi

    assert canvas_h > 0 and canvas_w > 0
    assert roi > 0
    assert roi_h <= canvas_h and roi_w <= canvas_w, \
        f"ROI {roi_h}x{roi_w} must be <= canvas {canvas_h}x{canvas_w}"
    assert args.n_rois_per_image >= 1
    assert 0.1 <= args.scale_min <= args.scale_max

    src_root = os.path.join(args.src_base, args.train_split)
    dst_root = os.path.join(args.src_base, args.dest_train_split)

    if args.overwrite and os.path.isdir(dst_root):
        shutil.rmtree(dst_root)
    ensure_dir(dst_root)

    pairs_by_cls = collect_pairs(src_root)
    if args.only_class:
        pairs_by_cls = {args.only_class: pairs_by_cls.get(args.only_class, [])}

    for cls, pairs in pairs_by_cls.items():
        if not pairs:
            print(f"[SKIP] {args.train_split} {cls}: empty")
            continue

        out_img_dir = os.path.join(dst_root, cls, "images")
        out_msk_dir = os.path.join(dst_root, cls, "masks")
        ensure_dir(out_img_dir)
        ensure_dir(out_msk_dir)

        use_pairs = pairs[:args.limit] if (args.limit and args.limit > 0) else pairs

        ok = 0
        empty_seed = 0

        for img_idx, (ip, mp) in enumerate(use_pairs):
            img = Image.open(ip).convert("RGB")
            msk = Image.open(mp).convert("L")

            # IMPORTANT: resize first to canvas (matches your training pipeline base)
            img_arr, msk_arr = resize_to_canvas(img, msk, canvas_h, canvas_w)

            H, W = msk_arr.shape
            stem = Path(ip).stem
            used_boxes = []

            for k in range(args.n_rois_per_image):
                best = None

                for _try in range(args.max_tries):
                    sc = float(rng.uniform(args.scale_min, args.scale_max))
                    rw = int(round(roi_w * sc))
                    rh = int(round(roi_h * sc))
                    rw = clamp(rw, 64, W)
                    rh = clamp(rh, 64, H)

                    seed = pick_seed_component(msk_arr, rng, min_comp_pixels=args.min_comp_pixels)
                    if seed is None:
                        empty_seed += 1
                        cx = rng.randint(0, W)
                        cy = rng.randint(0, H)
                    else:
                        cx, cy = seed

                    jx = int(rng.uniform(-args.jitter, args.jitter) * rw)
                    jy = int(rng.uniform(-args.jitter, args.jitter) * rh)
                    cx = int(np.clip(cx + jx, 0, W - 1))
                    cy = int(np.clip(cy + jy, 0, H - 1))

                    img_crop, msk_crop, box = crop_with_center(img_arr, msk_arr, cx, cy, rw, rh)

                    if used_boxes:
                        mx = max(bbox_iou(box, b) for b in used_boxes)
                        if mx > args.max_iou:
                            continue

                    ratio = float(msk_crop.sum()) / float(rw * rh)
                    if seed is None or ratio >= args.min_mask_ratio:
                        # normalize to fixed roi x roi for saving
                        if (rh, rw) != (roi_h, roi_w):
                            img_p = Image.fromarray(img_crop).resize((roi_w, roi_h), Image.BILINEAR)
                            msk_p = Image.fromarray((msk_crop * 255).astype(np.uint8)).resize((roi_w, roi_h), Image.NEAREST)
                            img_crop = np.asarray(img_p, np.uint8)
                            msk_crop = (np.asarray(msk_p, np.uint8) > 127).astype(np.uint8)
                        best = (img_crop, msk_crop, box)
                        break

                if best is None:
                    cx, cy = W // 2, H // 2
                    img_crop, msk_crop, box = crop_with_center(img_arr, msk_arr, cx, cy, roi_w, roi_h)
                else:
                    img_crop, msk_crop, box = best

                used_boxes.append(box)

                out_stem = f"{stem}__roi_{img_idx:05d}_{k:02d}"
                Image.fromarray(img_crop).save(os.path.join(out_img_dir, out_stem + ".png"))
                Image.fromarray(msk_crop * 255).save(os.path.join(out_msk_dir, out_stem + ".png"))
                ok += 1

        print(f"[OK] train ROI-from-resized | {cls}: {ok} ROI pairs "
              f"(from {len(use_pairs)} images x {args.n_rois_per_image}) | empty_seed={empty_seed}")

    print(f"\n[DONE] Saved train ROI to: {dst_root}")
    print("[NOTE] Validation stays untouched: you will evaluate using original val split (full images).")
    print("[TIP] For bbox workflow, keep ROI square (e.g., 384) and train model at the same square size.")

if __name__ == "__main__":
    main()
