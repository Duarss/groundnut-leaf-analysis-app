# app/utils/image_io.py
import io
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from app.core.config import Config


def _open_rgb(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _rotate90_if_portrait(img: Image.Image):
    """
    ROTATE_90 (counter-clockwise) hanya untuk kebutuhan model segmentasi (trained landscape).
    Return: (img_rotated_or_same, rotated_flag)
    """
    w, h = img.size
    if h > w:
        # portrait -> rotate CCW 90 agar jadi landscape
        return img.transpose(Image.ROTATE_90), True
    return img, False


def load_image_for_classification(image_bytes: bytes):
    """
    Bytes -> batch numpy (1, H, W, 3) preprocessed untuk EfficientNet-B4.

    IMPORTANT:
    - Sesuai revisi: klasifikasi TIDAK melakukan rotasi portrait->landscape.
    """
    img = _open_rgb(image_bytes)
    img = img.resize((Config.CLSF_IMG_W, Config.CLSF_IMG_H))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def load_image_for_segmentation(image_bytes: bytes, return_rotated: bool = False):
    """
    Bytes -> batch numpy (1, H, W, 3) float [0..1] untuk U-Net EfficientNetB0 encoder.

    Pipeline:
    - Jika portrait -> rotate 90 (CCW) agar sesuai pola training landscape
    - Resize ke (SEG_IMG_W, SEG_IMG_H) (default 640x480)
    - Normalisasi [0..1]

    return_rotated=True -> return (batch, rotated_flag)
    """
    img = _open_rgb(image_bytes)
    img, rotated = _rotate90_if_portrait(img)

    img = img.resize((Config.SEG_IMG_W, Config.SEG_IMG_H))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    if return_rotated:
        return arr, rotated
    return arr


def rotate_back_if_needed_pil(img: Image.Image, rotated: bool) -> Image.Image:
    """
    Jika sebelumnya kita ROTATE_90 (CCW), maka untuk kembali gunakan ROTATE_270 (CCW).
    """
    if not rotated:
        return img
    return img.transpose(Image.ROTATE_270)
