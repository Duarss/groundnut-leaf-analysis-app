import os
from PIL import Image
import imagehash

# =============== CONFIG ===============
# Set this to the folder you want to scan
FOLDER = r"C:\laragon\www\TA\groundnut-leaf-analysis-app\groundnut-backend\datasets\raw\ori_dataset"

# If True, also scan subfolders inside FOLDER
RECURSIVE = False

# Hamming distance threshold:
# 0  => almost identical
# 5  => similar (good starting point)
# 8-10 => more tolerant, catches more but may add a bit of noise
HASH_DISTANCE_THRESHOLD = 10
# ======================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def is_image_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTENSIONS


def iter_image_files(root_dir: str):
    """
    Yield full paths of image files under root_dir.
    Respects the RECURSIVE flag.
    """
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


def find_phash_duplicates_in_single_folder(folder: str):
    """
    Scan one folder for visually-duplicate images using pHash + Hamming distance.

    Logic:
    - First image that creates a group = reference.
    - Every new image is compared to all existing group references.
      - If min distance <= threshold -> duplicate of that group.
      - Else -> becomes a new reference group.
    Returns:
        groups: list of dicts with:
            {
                "ref_path": str,
                "ref_hash": imagehash.ImageHash,
                "dupes": [(dupe_path, distance), ...]
            }
    """
    groups = []  # list of {"ref_path", "ref_hash", "dupes": []}
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

        # Compare with all existing group references
        best_group = None
        best_distance = None

        for group in groups:
            dist = group["ref_hash"] - ph  # Hamming distance
            if best_distance is None or dist < best_distance:
                best_distance = dist
                best_group = group

        if best_group is not None and best_distance is not None and best_distance <= HASH_DISTANCE_THRESHOLD:
            # This image is a duplicate of the best group (closest reference)
            best_group["dupes"].append((img_path, best_distance))
        else:
            # No close reference found -> create a new group with this as reference
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

    groups = find_phash_duplicates_in_single_folder(FOLDER)

    # Filter only groups that actually have duplicates
    duplicate_groups = [g for g in groups if g["dupes"]]

    if not duplicate_groups:
        print(
            f"\n❌ No visually similar duplicate images found in this folder "
            f"(threshold = {HASH_DISTANCE_THRESHOLD})."
        )
    else:
        total_dupes = sum(len(g["dupes"]) for g in duplicate_groups)
        print(
            f"\n✅ Found {len(duplicate_groups)} reference image(s) "
            f"with duplicates (total duplicate files = {total_dupes}):\n"
        )

        for group in duplicate_groups:
            ref = group["ref_path"]
            dupes = group["dupes"]
            print(f"REFERENCE: {ref}")
            for dupe_path, dist in dupes:
                print(f"  DUPLICATE (dist={dist}): {dupe_path}")
            print("-" * 80)
