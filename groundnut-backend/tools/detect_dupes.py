# tools/detect_dupes.py
import os
from PIL import Image
import imagehash

LABEL_CLASS = "RUST"
FOLDER = rf"C:\laragon\www\TA\groundnut-leaf-analysis-app\groundnut-backend\datasets\raw\ori_dataset\{LABEL_CLASS}"
RECURSIVE = False
HASH_DISTANCE_THRESHOLD = 10
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
# ======================================

def is_image_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTENSIONS

def iter_image_files(root_dir: str):
    if RECURSIVE:
        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                if is_image_file(name):
                    yield os.path.join(dirpath, name)
    else:
        for name in os.listdir(root_dir):
            full_path = os.path.join(root_dir, name)
            if os.path.isfile(full_path) and is_image_file(name):
                yield full_path

def compute_phash(path: str):
    """
    Compute perceptual hash (pHash) of an image file.
    """
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return imagehash.phash(img)
    except Exception as e:
        print(f"[WARN] Failed to compute pHash for {path}: {e}")
        return None

def find_phash_duplicates(folder: str):
    groups = [] 
    total_files = 0

    print(f"\n=== Scanning folder for visually-duplicate images (pHash) ===")
    print(f"Folder: {folder}")
    print(f"Recursive: {RECURSIVE}")
    print(f"Hamming distance threshold: {HASH_DISTANCE_THRESHOLD}\n")

    for img_path in iter_image_files(folder):
        total_files += 1
        ph = compute_phash(img_path)
        if ph is None:
            continue

        best_group = None
        best_distance = None

        for group in groups:
            dist = group["ref_hash"] - ph
            if best_distance is None or dist < best_distance:
                best_distance = dist
                best_group = group

        if best_group is not None and best_distance is not None and best_distance <= HASH_DISTANCE_THRESHOLD:
            best_group["dupes"].append((img_path, best_distance))
        else:
            groups.append({
                "ref_path": img_path,
                "ref_hash": ph,
                "dupes": []
            })

    print(f"[INFO] Total image files scanned: {total_files}")
    print(f"[INFO] Total groups (unique-ish images): {len(groups)}")

    return groups

if __name__ == "__main__":
    if not os.path.isdir(FOLDER):
        print(f"ERROR: Folder not found: {FOLDER}")
        raise SystemExit(1)

    groups = find_phash_duplicates(FOLDER)

    duplicate_groups = [g for g in groups if g["dupes"]]

    if not duplicate_groups:
        print(
            f"\n No visually similar duplicate images found in this folder "
            f"(threshold = {HASH_DISTANCE_THRESHOLD})."
        )
    else:
        total_dupes = sum(len(g["dupes"]) for g in duplicate_groups)
        print(
            f"\n Found {len(duplicate_groups)} reference image(s) "
            f"with duplicates (total duplicate files = {total_dupes}):\n"
        )

        for group in duplicate_groups:
            ref = group["ref_path"]
            dupes = group["dupes"]
            print(f"REFERENCE: {ref}")
            for dupe_path, dist in dupes:
                print(f"  DUPLICATE (dist={dist}): {dupe_path}")
            print("-" * 80)
