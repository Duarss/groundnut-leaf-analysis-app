# tools/dataset_split.py
import os
import sys
import argparse
import random
import shutil
import csv
from pathlib import Path

import numpy as np
from PIL import Image

VALID_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
VALID_MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ---- optional GUI folder picker (if -i not provided) ----
def gui_pick_folder(title="Pilih folder dataset (cleaned_ori_dataset)"):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title=title)
        root.destroy()
        return path or None
    except Exception:
        return None

def parse_args():
    p = argparse.ArgumentParser(
        description="Split dataset dari cleaned_ori_dataset untuk classification atau segmentation (paired)."
    )
    p.add_argument("--task", "-t", type=str, required=True,
                   choices=["classification", "segmentation", "severity"],
                   help="Pilih mode split: classification | segmentation | severity")
    p.add_argument("--input", "-i", type=str,
                   help="Folder input cleaned_ori_dataset (berisi folder kelas, tiap kelas punya images/ dan masks/). "
                        "Jika tidak diisi, akan muncul dialog GUI.")
    p.add_argument("--output", "-o", type=str, required=True,
                   help="Folder output hasil split (classification_dataset / segmentation_dataset / severity_dataset).")
    p.add_argument("--ratio", "-r", type=float, nargs="+",
                   help="Rasio split. 2 angka untuk train,val | 3 angka untuk train,val,test (opsional). "
                        "Contoh: -r 0.7 0.3")
    p.add_argument("--seed", type=int, default=42, help="Seed untuk reproducibility (default: 42).")
    p.add_argument("--move", action="store_true",
                   help="Pindahkan file alih-alih menyalin (default: copy).")
    p.add_argument("--skip_healthy_for_seg", action="store_true", default=True,
                   help="(Default ON) Skip kelas HEALTHY untuk segmentation.")
    p.add_argument("--healthy_names", type=str, nargs="*", default=["HEALTHY", "Healthy", "healthy"],
                   help="Nama folder kelas yang dianggap HEALTHY.")
    p.add_argument("--min_val_per_class", type=int, default=8,
                   help="Peringatan jika jumlah per kelas di val < nilai ini (default: 8).")
    p.add_argument("--min_test_per_class", type=int, default=8,
                   help="Peringatan jika jumlah per kelas di test < nilai ini (default: 8).")
    p.add_argument("--orphans_dir", type=str, default="orphans_report",
                   help="Folder untuk menyimpan laporan/penampungan data tanpa pasangan (khusus segmentation).")

    # ===== Stratified split (segmentation only) =====
    p.add_argument("--stratify_lesion", action="store_true",
                   help="(Segmentation only) Stratified split berdasarkan ukuran lesi per kelas.")

    p.add_argument("--stratify_mode", type=str, default="quantile",
               choices=["fixed", "quantile"],
               help="Mode stratify: fixed (tiny/mid/large via threshold) atau quantile (3 bucket berbasis quantile).")

    # threshold mode params (legacy)
    p.add_argument("--tiny_thr", type=float, default=0.002,
                   help="(threshold mode) Threshold area ratio untuk bucket tiny (default 0.002 = 0.2%).")
    p.add_argument("--mid_thr", type=float, default=0.02,
                   help="(threshold mode) Threshold area ratio untuk bucket mid (default 0.02 = 2%).")

    # quantile mode params (recommended)
    p.add_argument("--q1", type=float, default=0.33,
                   help="(quantile mode) batas quantile pertama (default 0.33)")
    p.add_argument("--q2", type=float, default=0.66,
                   help="(quantile mode) batas quantile kedua (default 0.66)")

    p.add_argument("--min_bucket", type=int, default=6,
                   help="Minimal jumlah sample per bucket agar stratify jalan; jika < ini maka fallback random split.")

    return p.parse_args()

def safe_reset_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(dirpath: Path):
    if not dirpath.exists():
        return []
    files = []
    for f in dirpath.iterdir():
        if f.is_file() and f.suffix.lower() in VALID_IMG_EXTS:
            files.append(f)
    return sorted(files)

def find_mask_for_image(mask_dir: Path, img_path: Path):
    """Mask ditentukan berdasarkan stem yang sama, extension boleh beda."""
    stem = img_path.stem
    for ext in VALID_MASK_EXTS:
        cand = mask_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None

def copy_or_move(src: Path, dst: Path, move: bool):
    ensure_dir(dst.parent)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

def split_list(items, ratio, rng: random.Random):
    """Return dict split_name -> list items. ratio len 2 or 3."""
    items = list(items)
    rng.shuffle(items)
    n = len(items)

    if len(ratio) == 2:
        n_train = int(n * ratio[0])
        train = items[:n_train]
        val = items[n_train:]
        return {"train": train, "val": val}
    else:
        n_train = int(n * ratio[0])
        n_val = int(n * ratio[1])
        train = items[:n_train]
        val = items[n_train:n_train + n_val]
        test = items[n_train + n_val:]
        return {"train": train, "val": val, "test": test}

# ========================= lesion size helpers =========================
def mask_area_ratio(mask_path: Path) -> float:
    """Return mean(mask_bin) as lesion area ratio."""
    m = Image.open(mask_path).convert("L")
    arr = np.asarray(m, dtype=np.uint8)
    binm = (arr >= 128).astype(np.float32)
    return float(binm.mean())

def bucket_by_ratio_threshold(r: float, tiny_thr: float, mid_thr: float) -> str:
    if r < tiny_thr:
        return "tiny"
    if r < mid_thr:
        return "mid"
    return "large"

def stratified_split_pairs_by_threshold(paired, ratio, rng, tiny_thr, mid_thr, min_bucket):
    """
    Stratify by fixed thresholds: tiny/mid/large.
    Fallback random if buckets not meaningful.
    """
    buckets = {"tiny": [], "mid": [], "large": []}
    for (img, msk) in paired:
        try:
            r = mask_area_ratio(msk)
        except Exception:
            r = 0.01
        b = bucket_by_ratio_threshold(r, tiny_thr, mid_thr)
        buckets[b].append((img, msk))

    sizes = {k: len(v) for k, v in buckets.items()}
    non_empty = [k for k, v in sizes.items() if v > 0]

    if len(non_empty) < 2 or any(sizes[k] < min_bucket for k in non_empty):
        return split_list(paired, ratio, rng), {"mode": "random_fallback", "bucket_sizes": sizes}

    parts = {k: [] for k in ("train", "val", "test")}
    meta = {"mode": "stratified", "bucket_sizes": sizes}

    for bname, items in buckets.items():
        if not items:
            continue
        sub = split_list(items, ratio, rng)
        for split_name, lst in sub.items():
            parts[split_name].extend(lst)

    for split_name in list(parts.keys()):
        if not parts[split_name]:
            parts.pop(split_name, None)
            continue
        rng.shuffle(parts[split_name])

    return parts, meta

def stratified_split_pairs_by_quantile(paired, ratio, rng, q1=0.33, q2=0.66, min_bucket=6):
    """
    Stratify by per-class area quantiles.
    Buckets: small(<q1v), mid(<q2v), large(>=q2v)
    """
    ratios = []
    for (_, msk) in paired:
        try:
            r = mask_area_ratio(msk)
        except Exception:
            r = 0.01
        ratios.append(r)

    if len(ratios) == 0:
        return split_list(paired, ratio, rng), {"mode": "random_fallback", "bucket_sizes": {"small": 0, "mid": 0, "large": 0}}

    rnp = np.asarray(ratios, dtype=np.float32)
    q1v = float(np.quantile(rnp, float(q1)))
    q2v = float(np.quantile(rnp, float(q2)))

    buckets = {"small": [], "mid": [], "large": []}
    for (img, msk), r in zip(paired, ratios):
        if r < q1v:
            buckets["small"].append((img, msk))
        elif r < q2v:
            buckets["mid"].append((img, msk))
        else:
            buckets["large"].append((img, msk))

    sizes = {k: len(v) for k, v in buckets.items()}
    non_empty = [k for k, v in sizes.items() if v > 0]

    if len(non_empty) < 2 or any(sizes[k] < min_bucket for k in non_empty):
        return split_list(paired, ratio, rng), {
            "mode": "random_fallback",
            "bucket_sizes": sizes,
            "q1v": q1v,
            "q2v": q2v
        }

    parts = {k: [] for k in ("train", "val", "test")}
    meta = {"mode": "stratified", "bucket_sizes": sizes, "q1v": q1v, "q2v": q2v}

    for bname, items in buckets.items():
        if not items:
            continue
        sub = split_list(items, ratio, rng)
        for split_name, lst in sub.items():
            parts[split_name].extend(lst)

    for split_name in list(parts.keys()):
        if not parts[split_name]:
            parts.pop(split_name, None)
            continue
        rng.shuffle(parts[split_name])

    return parts, meta

# ========================= summarize helpers =========================
def count_files_flat(class_dir: Path):
    if not class_dir.exists():
        return 0
    return sum(1 for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_IMG_EXTS)

def summarize_split_classification(root: Path, min_val=8, min_test=8):
    for split in ("train", "val", "test"):
        base = root / split
        if not base.exists():
            print(f"\n📋 {split.upper()}: (tidak ada)")
            continue
        classes = [d for d in sorted(base.iterdir()) if d.is_dir()]
        if not classes:
            print(f"\n📋 {split.upper()}: (kosong)")
            continue

        counts = {d.name: count_files_flat(d) for d in classes}
        total = sum(counts.values())
        width = max((len(k) for k in counts), default=10)

        print(f"\n📋 {split.upper()} summary (total {total}):")
        for k, v in counts.items():
            warn = ""
            if split == "val" and v < min_val:
                warn = "  ⚠️ < min_val_per_class"
            if split == "test" and v < min_test:
                warn = "  ⚠️ < min_test_per_class"
            print(f"  {k:<{width}} : {v:5d}{warn}")

def count_pairs_seg(class_dir: Path):
    img_dir = class_dir / "images"
    if not img_dir.exists():
        return 0
    return sum(1 for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_IMG_EXTS)

def summarize_split_segmentation(root: Path, min_val=8, min_test=8):
    for split in ("train", "val", "test"):
        base = root / split
        if not base.exists():
            print(f"\n📋 {split.upper()}: (tidak ada)")
            continue
        classes = [d for d in sorted(base.iterdir()) if d.is_dir()]
        if not classes:
            print(f"\n📋 {split.upper()}: (kosong)")
            continue

        counts = {d.name: count_pairs_seg(d) for d in classes}
        total = sum(counts.values())
        width = max((len(k) for k in counts), default=10)

        print(f"\n📋 {split.upper()} summary (total pairs {total}):")
        for k, v in counts.items():
            warn = ""
            if split == "val" and v < min_val:
                warn = "  ⚠️ < min_val_per_class"
            if split == "test" and v < min_test:
                warn = "  ⚠️ < min_test_per_class"
            print(f"  {k:<{width}} : {v:5d}{warn}")

# ========================= split modes =========================
def split_classification(input_root: Path, output_root: Path, ratio, seed: int, move: bool):
    rng = random.Random(seed)
    safe_reset_dir(output_root)

    classes = [d for d in sorted(input_root.iterdir()) if d.is_dir()]
    if not classes:
        raise RuntimeError(f"Tidak ada folder kelas di: {input_root}")

    print("\n=== MODE: CLASSIFICATION ===")
    for cls_dir in classes:
        cls = cls_dir.name
        img_dir = cls_dir / "images"
        if not img_dir.exists():
            print(f"⚠️  Lewati '{cls}' karena tidak ada folder images/: {img_dir}")
            continue

        imgs = list_images(img_dir)
        if not imgs:
            print(f"⚠️  Lewati '{cls}' karena tidak ada file gambar valid di: {img_dir}")
            continue

        parts = split_list(imgs, ratio, rng)

        for split_name, files in parts.items():
            out_cls = output_root / split_name / cls
            ensure_dir(out_cls)
            for f in files:
                dst = out_cls / f.name
                copy_or_move(f, dst, move)

        print(f"✅ {cls}: total={len(imgs)} | " +
              " | ".join([f"{k}={len(v)}" for k, v in parts.items()]))

    summarize_split_classification(output_root)

def split_segmentation(input_root: Path, output_root: Path, ratio, seed: int, move: bool,
                       skip_healthy: bool, healthy_names, orphans_dir_name: str,
                       stratify_lesion: bool, stratify_mode: str,
                       tiny_thr: float, mid_thr: float,
                       q1: float, q2: float,
                       min_bucket: int):
    rng = random.Random(seed)

    safe_reset_dir(output_root)

    orphans_root = Path(orphans_dir_name)
    safe_reset_dir(orphans_root)
    (orphans_root / "no_mask").mkdir(parents=True, exist_ok=True)
    (orphans_root / "no_image").mkdir(parents=True, exist_ok=True)

    classes = [d for d in sorted(input_root.iterdir()) if d.is_dir()]
    if not classes:
        raise RuntimeError(f"Tidak ada folder kelas di: {input_root}")

    print("\n=== MODE: SEGMENTATION (PAIRED) ===")
    if stratify_lesion:
        if stratify_mode == "quantile":
            print(f"✅ Stratified by lesion size (quantile): q1={q1}, q2={q2}, min_bucket={min_bucket}")
        else:
            print(f"✅ Stratified by lesion size (threshold): tiny_thr={tiny_thr}, mid_thr={mid_thr}, min_bucket={min_bucket}")

    for cls_dir in classes:
        cls = cls_dir.name

        if skip_healthy and cls in set(healthy_names):
            print(f"⏭️  Skip '{cls}' (HEALTHY tidak ikut segmentasi)")
            continue

        img_dir = cls_dir / "images"
        mask_dir = cls_dir / "masks"

        if not img_dir.exists():
            print(f"⚠️  Lewati '{cls}' karena tidak ada folder images/: {img_dir}")
            continue
        if not mask_dir.exists():
            print(f"⚠️  Lewati '{cls}' karena tidak ada folder masks/: {mask_dir}")
            continue

        imgs = list_images(img_dir)
        if not imgs:
            print(f"⚠️  Lewati '{cls}' karena tidak ada file gambar valid di: {img_dir}")
            continue

        paired = []
        for img in imgs:
            m = find_mask_for_image(mask_dir, img)
            if m is None:
                copy_or_move(img, orphans_root / "no_mask" / cls / img.name, move=False)
                continue
            paired.append((img, m))

        img_stems = {p.stem for p in imgs}
        for m in mask_dir.iterdir():
            if m.is_file() and m.suffix.lower() in VALID_MASK_EXTS and m.stem not in img_stems:
                copy_or_move(m, orphans_root / "no_image" / cls / m.name, move=False)

        if not paired:
            print(f"⚠️  '{cls}' tidak punya pasangan image-mask yang valid.")
            continue

        # ---- stratified split (optional) ----
        if stratify_lesion:
            if stratify_mode == "quantile":
                parts, meta = stratified_split_pairs_by_quantile(
                    paired=paired,
                    ratio=ratio,
                    rng=rng,
                    q1=q1,
                    q2=q2,
                    min_bucket=min_bucket
                )
                if meta["mode"] == "stratified":
                    bs = meta["bucket_sizes"]
                    print(f"🧪 {cls}: quantile buckets small={bs['small']}, mid={bs['mid']}, large={bs['large']} | q1v={meta['q1v']:.6f} q2v={meta['q2v']:.6f}")
                else:
                    bs = meta["bucket_sizes"]
                    print(f"🟡 {cls}: fallback random (quantile bucket kecil) small={bs['small']}, mid={bs['mid']}, large={bs['large']} | q1v={meta.get('q1v',0):.6f} q2v={meta.get('q2v',0):.6f}")
            else:
                parts, meta = stratified_split_pairs_by_threshold(
                    paired=paired,
                    ratio=ratio,
                    rng=rng,
                    tiny_thr=tiny_thr,
                    mid_thr=mid_thr,
                    min_bucket=min_bucket
                )
                if meta["mode"] == "stratified":
                    bs = meta["bucket_sizes"]
                    print(f"🧪 {cls}: threshold buckets tiny={bs['tiny']}, mid={bs['mid']}, large={bs['large']}")
                else:
                    bs = meta["bucket_sizes"]
                    print(f"🟡 {cls}: fallback random (bucket kecil) tiny={bs['tiny']}, mid={bs['mid']}, large={bs['large']}")
        else:
            parts = split_list(paired, ratio, rng)

        for split_name, pairs in parts.items():
            out_img_dir = output_root / split_name / cls / "images"
            out_mask_dir = output_root / split_name / cls / "masks"
            ensure_dir(out_img_dir)
            ensure_dir(out_mask_dir)

            for (img, msk) in pairs:
                copy_or_move(img, out_img_dir / img.name, move)
                copy_or_move(msk, out_mask_dir / msk.name, move)

        print(f"✅ {cls}: paired_total={len(paired)} | " +
              " | ".join([f"{k}={len(v)}" for k, v in parts.items()]))

    summarize_split_segmentation(output_root)

    print(f"\n🧾 Orphans report disimpan di: {orphans_root}")
    print(f"   - no_mask  : image yang tidak punya pasangan mask")
    print(f"   - no_image : mask yang tidak punya pasangan image (audit)")

def split_severity(input_root: Path, output_root: Path, ratio, seed: int, move: bool):
    """
    PAIR SPLIT (image+mask) + BALANCED PER CLASS (stratified) untuk layout FLAT:

    INPUT:
      {ROOT}/images
      {ROOT}/masks

    OUTPUT:
      {output_root}/train/images + masks
      {output_root}/val/images   + masks
      (optional) test/...

    Kelas diambil dari NAMA FILE (image/mask) berdasarkan substring match.
    """

    rng = random.Random(seed)

    img_dir = input_root / "images"
    mask_dir = input_root / "masks"
    if not img_dir.exists() or not mask_dir.exists():
        raise RuntimeError("Folder images/ atau masks/ tidak ditemukan pada input_root (layout flat).")

    images = list_images(img_dir)
    if not images:
        raise RuntimeError("Tidak ada file gambar valid di folder images/")

    # ---- build pairs (only those with matching masks) ----
    pairs = []
    for img in images:
        m = find_mask_for_image(mask_dir, img)
        if m is not None:
            pairs.append((img, m))

    if not pairs:
        raise RuntimeError("Tidak ada pasangan image-mask yang valid")

    # ---- class names (as provided) ----
    CLASS_NAMES = [
        "HEALTHY",
        "ALTERNARIA LEAF SPOT",
        "LEAF SPOT (EARLY AND LATE)",
        "ROSETTE",
        "RUST",
    ]

    # ---- normalization helpers (case-insensitive, unify separators) ----
    def _norm(s: str) -> str:
        s = s.lower()
        # normalize separators to space
        for ch in ["_", "-", ".", ",", ";", ":", "+", "|", "/"]:
            s = s.replace(ch, " ")
        # keep parentheses as-is (since your label contains them), but normalize multiple spaces
        s = " ".join(s.split())
        return s

    norm_classes = [(c, _norm(c)) for c in CLASS_NAMES]
    # Prefer longest match to avoid accidental partial collisions
    norm_classes.sort(key=lambda x: len(x[1]), reverse=True)

    def get_class_from_name(p: Path) -> str:
        name = _norm(p.stem)
        hits = []
        for orig, nc in norm_classes:
            if nc in name:
                hits.append((orig, len(nc)))
        if not hits:
            return None
        # pick the longest matched class string
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[0][0]

    # ---- stratify pairs by class ----
    by_class = {c: [] for c in CLASS_NAMES}
    unknown = []

    for (img, msk) in pairs:
        cls_img = get_class_from_name(img)
        cls_msk = get_class_from_name(msk)

        # if both present but disagree, treat as error (data issue)
        if cls_img and cls_msk and cls_img != cls_msk:
            raise RuntimeError(
                f"Label mismatch antara image dan mask:\n"
                f"  image: {img.name} -> {cls_img}\n"
                f"  mask : {msk.name} -> {cls_msk}"
            )

        cls = cls_img or cls_msk
        if cls is None:
            unknown.append(img.name)
            continue

        by_class[cls].append((img, msk))

    if unknown:
        ex = ", ".join(unknown[:10])
        raise RuntimeError(
            f"Ada {len(unknown)} pasangan yang tidak bisa dipetakan ke kelas dari nama file. "
            f"Contoh: {ex}\n"
            f"Pastikan nama file mengandung salah satu kelas: {CLASS_NAMES}"
        )

    # ---- compute per-class counts (must be equal per your requirement) ----
    if len(ratio) not in (2, 3):
        raise RuntimeError("Rasio harus 2 angka (train,val) atau 3 angka (train,val,test).")

    def per_class_counts(n: int, ratio_tuple):
        # untuk n=20 & 0.7 -> 14, sisa 6
        if len(ratio_tuple) == 2:
            n_train = int(round(n * float(ratio_tuple[0])))
            n_train = max(0, min(n, n_train))
            return {"train": n_train, "val": n - n_train}
        else:
            n_train = int(round(n * float(ratio_tuple[0])))
            n_val = int(round(n * float(ratio_tuple[1])))
            n_train = max(0, min(n, n_train))
            n_val = max(0, min(n - n_train, n_val))
            return {"train": n_train, "val": n_val, "test": n - n_train - n_val}

    # validate balanced requirement: each class must be able to contribute equally
    per_cls_n = {c: len(by_class[c]) for c in CLASS_NAMES}
    # if you truly expect 20 each, enforce it so result is exactly 14/6
    # (kalau kamu mau fleksibel, bagian ini bisa dilonggarkan)
    expected_each = 20
    bad = {c: n for c, n in per_cls_n.items() if n != expected_each}
    if bad:
        raise RuntimeError(
            "Jumlah pair per kelas tidak sesuai ekspektasi (harus 20 per kelas agar jadi 14/6).\n"
            + "\n".join([f"  - {c}: {n}" for c, n in bad.items()])
        )

    cnt = per_class_counts(expected_each, ratio)
    # enforce exact counts as you asked
    if len(ratio) == 2:
        if cnt["train"] != 14 or cnt["val"] != 6:
            raise RuntimeError(f"Rasio tidak menghasilkan 14/6 per kelas. Hasil: {cnt}")
    else:
        # kalau 3-way split, kamu bisa atur sendiri; di sini tidak dipaksa 14/6
        pass

    # ---- make final parts ----
    parts = {k: [] for k in ("train", "val", "test")}
    report = {}

    for cls in CLASS_NAMES:
        items = list(by_class[cls])
        rng.shuffle(items)

        start = 0
        cls_report = {}
        for split_name in ("train", "val", "test"):
            if split_name not in cnt:
                continue
            take = cnt[split_name]
            chunk = items[start:start + take]
            start += take
            cls_report[split_name] = len(chunk)
            parts[split_name].extend([(cls, img, msk) for (img, msk) in chunk])

        report[cls] = cls_report

    # drop empty split (e.g. no test)
    for k in list(parts.keys()):
        if not parts[k]:
            parts.pop(k, None)

    # ---- write output ----
    safe_reset_dir(output_root)

    print("\n=== SEVERITY PAIRED SPLIT (FLAT + BALANCED PER CLASS) ===")
    total_pairs = sum(len(v) for v in parts.values())
    print(f"✅ Total paired samples : {len(pairs)}")
    for cls in CLASS_NAMES:
        rep = report[cls]
        rep_str = " | ".join([f"{k}={rep[k]}" for k in ("train", "val", "test") if k in rep])
        print(f"  - {cls}: total={len(by_class[cls])} | {rep_str}")

    for split_name, items in parts.items():
        out_img = output_root / split_name / "images"
        out_msk = output_root / split_name / "masks"
        ensure_dir(out_img)
        ensure_dir(out_msk)

        for idx, (cls, img, msk) in enumerate(items):
            # output stem sama agar pasangan tetap nyambung
            dst_stem = f"{idx:07d}_{cls.replace(' ', '_')}_{img.stem}"
            dst_img = out_img / f"{dst_stem}{img.suffix.lower()}"
            dst_msk = out_msk / f"{dst_stem}{msk.suffix.lower()}"
            copy_or_move(img, dst_img, move)
            copy_or_move(msk, dst_msk, move)

        print(f"✅ {split_name:<5}: {len(items)} pairs -> {output_root / split_name}")


def main():
    args = parse_args()

    if not args.input:
        title = "Pilih folder dataset (cleaned_ori_dataset)"
        if args.task == "severity":
            title = "Pilih ROOT dataset (5 kelas; masing-masing punya images/ dan masks/ lesion GT)"
        input_folder = gui_pick_folder(title=title)
    else:
        input_folder = args.input

    if not input_folder:
        print("❌ Folder input tidak diberikan.")
        sys.exit(1)

    input_root = Path(input_folder)
    if not input_root.exists() or not input_root.is_dir():
        print(f"❌ Folder input tidak valid: {input_root}")
        sys.exit(1)

    if not args.ratio:
        ratio = (0.7, 0.3)
    else:
        ratio = tuple(args.ratio)

    if len(ratio) not in (2, 3):
        print("❌ Rasio harus 2 angka (train,val) atau 3 angka (train,val,test).")
        sys.exit(1)

    if abs(sum(ratio) - 1.0) > 1e-6:
        print(f"❌ Jumlah rasio harus 1.0, saat ini = {sum(ratio):.4f}")
        sys.exit(1)

    output_root = Path(args.output)

    print("=== Konfigurasi Split ===")
    if len(ratio) == 2:
        print(f"Rasio  : train={ratio[0]:.2f}, val={ratio[1]:.2f}")
    else:
        print(f"Rasio  : train={ratio[0]:.2f}, val={ratio[1]:.2f}, test={ratio[2]:.2f}")
    print(f"Task   : {args.task}")
    print(f"Input  : {input_root}")
    print(f"Output : {output_root}")
    print(f"Seed   : {args.seed}")
    print(f"Mode   : {'move' if args.move else 'copy'}")
    if args.task == "segmentation":
        print(f"Stratify lesion: {args.stratify_lesion}")
        if args.stratify_lesion:
            print(f"Stratify mode : {args.stratify_mode}")
            if args.stratify_mode == "quantile":
                print(f"Quantiles     : q1={args.q1}, q2={args.q2}, min_bucket={args.min_bucket}")
            else:
                print(f"Thresholds    : tiny_thr={args.tiny_thr}, mid_thr={args.mid_thr}, min_bucket={args.min_bucket}")
    print("=========================")

    try:
        if args.task == "classification":
            split_classification(
                input_root=input_root,
                output_root=output_root,
                ratio=ratio,
                seed=args.seed,
                move=args.move
            )
        elif args.task == "segmentation":
            split_segmentation(
                input_root=input_root,
                output_root=output_root,
                ratio=ratio,
                seed=args.seed,
                move=args.move,
                skip_healthy=args.skip_healthy_for_seg,
                healthy_names=args.healthy_names,
                orphans_dir_name=args.orphans_dir,
                stratify_lesion=args.stratify_lesion,
                stratify_mode=args.stratify_mode,
                tiny_thr=args.tiny_thr,
                mid_thr=args.mid_thr,
                q1=args.q1,
                q2=args.q2,
                min_bucket=args.min_bucket
            )
            print("\nℹ️ Next:")
            print("   - Jalankan preprocessing segmentasi (paired augmentation) untuk membuat train_balanced/ dan val_balanced/")
        elif args.task == "severity":
            split_severity(
                input_root=input_root,
                output_root=output_root,
                ratio=ratio,
                seed=args.seed,
                move=args.move
            )
            print("\nℹ️ Next:")
            print("   - Jalankan severity_preprocessing.py untuk generate leaf masks (train/val)")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
