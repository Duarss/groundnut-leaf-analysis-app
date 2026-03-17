# tools/classification_preprocessing.py
import os
import random
import shutil
from PIL import Image
from tqdm import tqdm
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from statistics import median
import hashlib

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

SOURCE_DIR = "datasets/processed/classification_dataset"
TARGET_DIR = "datasets/processed/classification_dataset"

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ===================== Prinsip ilmiah balancing =====================
R_MAX = 5
MIN_TRAIN_TARGET = 80
MAX_TRAIN_TARGET = 1000

VAL_BALANCING = True
VAL_BALANCING_MODE = "downsample"
VAL_TARGET_PER_CLASS = None

# ===================== Anti-duplicate augmentation gate =====================
FINGERPRINT_SIZE = (32, 32)
MAX_TRIES_PER_AUG = 25       
DUP_SKIP_LOG_EVERY = 200      

def image_fingerprint(pil_img: Image.Image) -> str:
    im = pil_img.convert("L").resize(FINGERPRINT_SIZE, Image.BILINEAR)
    arr = np.asarray(im, dtype=np.uint8)
    arr = (arr - arr.mean()).astype(np.int16)
    arr_bytes = arr.tobytes()
    return hashlib.md5(arr_bytes).hexdigest()

# ===================== Augmentasi TRAIN =====================
augmentor_train = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=0.10,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    channel_shift_range=15.0,
    fill_mode="reflect"
)

# ===================== Utilities =====================
def safe_reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def list_dirs(path):
    blacklist = {"train", "val", "train_balanced"}
    dirs = []
    if not os.path.exists(path):
        return dirs
    for d in sorted(os.listdir(path)):
        full = os.path.join(path, d)
        if os.path.isdir(full) and d not in blacklist and not d.startswith(".") and not d.startswith("_"):
            dirs.append(d)
    return dirs

def list_images(path):
    if not os.path.exists(path):
        return []
    return [f for f in sorted(os.listdir(path)) if f.lower().endswith(VALID_EXTS)]

def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

def deterministic_sample(imgs, k, class_name, base_seed=GLOBAL_SEED):
    rng = random.Random(base_seed + sum(map(ord, class_name)))
    items = sorted(imgs)
    rng.shuffle(items)
    return items[:k]

def print_class_summary(root, split_name):
    total = 0
    print(f"\n{split_name.upper()}_BALANCED summary:")
    if not os.path.exists(root):
        print("  (folder tidak ditemukan)")
        return
    for cls in sorted(list_dirs(root)):
        n = len(list_images(os.path.join(root, cls)))
        total += n
        print(f"  {cls:<27}: {n:5d}")
    print(f"  Total{'':23}: {total:5d}")

# ===================== Hitung target balance TRAIN =====================
def compute_train_target(train_dir):
    class_dirs = list_dirs(train_dir)
    if not class_dirs:
        raise RuntimeError(f"Tidak ada folder kelas valid di: {train_dir}")

    counts = {}
    for cls in class_dirs:
        cdir = os.path.join(train_dir, cls)
        counts[cls] = len(list_images(cdir))

    values = list(counts.values())
    n_min = min(values)
    n_med = int(median(values))

    raw_target = min(n_med, int(R_MAX * n_min))
    target = max(MIN_TRAIN_TARGET, min(MAX_TRAIN_TARGET, raw_target))

    print("\n=== Penentuan Target Balanced TRAIN (ilmiah & reproducible) ===")
    print(f"Train counts per class : {counts}")
    print(f"n_min (kelas paling sedikit)      = {n_min}")
    print(f"median(n_k) (ukuran 'wajar')      = {n_med}")
    print(f"R_MAX (cap augment multiplier)    = {R_MAX}")
    print(f"raw_target = min({n_med}, {R_MAX}*{n_min}) = {raw_target}")
    if target != raw_target:
        print(f"clamp => final_target = {target} (dibatasi MIN/MAX)")
    else:
        print(f"final_target = {target}")
    print("=============================================================\n")

    return target

# ===================== Core processing =====================
def process_train_with_augment():
    split = "train"
    source_dir = os.path.join(SOURCE_DIR, split)
    target_dir = os.path.join(TARGET_DIR, f"{split}_balanced")

    if not os.path.exists(source_dir):
        print(f"[SKIP] Folder sumber split '{split}' tidak ditemukan: {source_dir}")
        return

    print(f"\n=== Processing {split.upper()} split (AUGMENT & BALANCE) ===")
    safe_reset_dir(target_dir)

    class_dirs = list_dirs(source_dir)
    if not class_dirs:
        print(f"Tidak ditemukan folder kelas yang valid di: {source_dir}")
        return

    target_count = compute_train_target(source_dir)

    for class_name in class_dirs:
        class_path = os.path.join(source_dir, class_name)
        target_class_path = os.path.join(target_dir, class_name)
        os.makedirs(target_class_path, exist_ok=True)

        image_files = list_images(class_path)
        current_count = len(image_files)

        if current_count == 0:
            print(f"Lewati '{class_name}' ({split}): tidak ada file gambar yang valid.")
            continue

        print(f"\nKelas '{class_name}' (train): {current_count} → target {target_count}")

        existing_fps = set()
        for img_name in image_files:
            try:
                im = Image.open(os.path.join(class_path, img_name)).convert("RGB")
                existing_fps.add(image_fingerprint(im))
            except Exception:
                pass

        if current_count > target_count:
            chosen = deterministic_sample(image_files, target_count, class_name, base_seed=GLOBAL_SEED)
            for img_name in tqdm(chosen, desc=f"Copy {class_name}", ncols=80):
                copy_file(os.path.join(class_path, img_name),
                          os.path.join(target_class_path, img_name))
            continue

        for img_name in tqdm(image_files, desc=f"Copy {class_name}", ncols=80):
            copy_file(os.path.join(class_path, img_name),
                      os.path.join(target_class_path, img_name))

        if current_count < target_count:
            needed = target_count - current_count
            augmented = 0
            skipped_dup = 0
            print(f"   Augmenting {class_name}: butuh {needed} tambahan...")

            base_list = deterministic_sample(image_files, len(image_files), class_name, base_seed=GLOBAL_SEED)
            i = 0

            while augmented < needed:
                base_img_name = base_list[i % len(base_list)]
                i += 1
                img_path = os.path.join(class_path, base_img_name)

                img = Image.open(img_path).convert("RGB")
                img_array = np.expand_dims(np.array(img), axis=0)

                tries = 0
                saved = False
                while tries < MAX_TRIES_PER_AUG and not saved:
                    dyn_seed = (GLOBAL_SEED
                                + (sum(map(ord, class_name)) * 100000)
                                + (augmented * 9973)
                                + tries)

                    for batch in augmentor_train.flow(img_array, batch_size=1, seed=dyn_seed):
                        aug_img = Image.fromarray(batch[0].astype("uint8"))

                        fp = image_fingerprint(aug_img)
                        if fp in existing_fps:
                            skipped_dup += 1
                            tries += 1
                            if skipped_dup % DUP_SKIP_LOG_EVERY == 0:
                                print(f"   ... skipped duplicates so far: {skipped_dup}")
                            break

                        existing_fps.add(fp)
                        base = os.path.splitext(base_img_name)[0].replace(" ", "_")
                        aug_name = f"aug_{augmented:05d}_{base}.jpg"
                        aug_img.save(os.path.join(target_class_path, aug_name), quality=95)
                        augmented += 1
                        saved = True
                        break

                if not saved:
                    base = os.path.splitext(base_img_name)[0].replace(" ", "_")
                    aug_name = f"aug_fallback_{augmented:05d}_{base}.jpg"
                    img.save(os.path.join(target_class_path, aug_name), quality=95)
                    augmented += 1

            print(f"Done augment '{class_name}': augmented={augmented}, skipped_dup={skipped_dup}, MAX_TRIES={MAX_TRIES_PER_AUG}")

    print(f"\nSplit 'train' selesai di-balance dan disimpan di '{target_dir}'.")
    print_class_summary(target_dir, "train")

def augment_and_balance_dataset(split):
    if split == "train":
        process_train_with_augment()
    else:
        print(f"[SKIP] Split '{split}' tidak dikenali.")

if __name__ == "__main__":
    for split in ["train", "val"]:
        augment_and_balance_dataset(split)
    print("\nSemua proses selesai.")
