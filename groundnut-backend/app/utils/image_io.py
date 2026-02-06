# app/utils/image_io.py
import io
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.efficientnet import preprocess_input
from app.core.config import Config

# Optional HEIC support (iOS):
# pip install pillow-heif
_HEIF_READY = False
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
    _HEIF_READY = True
except Exception:
    _HEIF_READY = False


def _open_rgb(image_bytes: bytes) -> Image.Image:
    """
    Buka image bytes -> PIL RGB.
    Support HEIC kalau pillow-heif terpasang.
    """
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as e:
        # Kalau file HEIC tapi pillow-heif belum terpasang
        if not _HEIF_READY:
            raise ValueError(
                "File gambar tidak dikenali. Jika ini .HEIC dari iPhone, "
                "install 'pillow-heif' di backend agar bisa diproses."
            ) from e
        # Kalau sudah register tapi masih gagal, lempar error jelas
        raise ValueError("File gambar tidak dapat dibuka (format tidak didukung / corrupt).") from e


def load_image_for_classification(image_bytes: bytes):
    """
    Bytes -> batch numpy (1, H, W, 3) preprocessed untuk EfficientNet-B4.
    """
    img = _open_rgb(image_bytes)
    img = img.resize((Config.CLSF_IMG_W, Config.CLSF_IMG_H))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def load_image_for_segmentation(image_bytes: bytes):
    """
    Bytes -> batch numpy (1, H, W, 3) float [0..1] untuk U-Net encoder EfficientNet-B0.
    """
    img = _open_rgb(image_bytes)
    img = img.resize((Config.SEG_IMG_W, Config.SEG_IMG_H))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr
