# app/utils/image_io.py
import io
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from app.core.config import Config

def _open_rgb(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def _ensure_landscape(img: Image.Image) -> Image.Image:
    """
    Jika image portrait (H > W), rotate 90 derajat ke landscape.
    Landscape dibiarkan apa adanya.
    """
    w, h = img.size
    if h > w:
        return img.rotate(90, expand=True)
    return img

def load_image_for_classification(image_bytes: bytes):
    """
    Bytes -> batch numpy (1, H, W, 3) preprocessed untuk EfficientNet-B4.
    """
    img = _open_rgb(image_bytes)
    img = _ensure_landscape(img)
    img = img.resize((Config.CLSF_IMG_W, Config.CLSF_IMG_H))

    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def load_image_for_segmentation(image_bytes: bytes):
    """
    Bytes -> batch numpy (1, H, W, 3) preprocessed untuk U-Net dengan encoder EfficientNet-B0.
    """
    img = _open_rgb(image_bytes)
    img = _ensure_landscape(img)
    img = img.resize((Config.SEG_IMG_W, Config.SEG_IMG_H))

    arr = np.array(img, dtype=np.float32) / 255.0  # Normalisasi ke [0, 1]
    arr = np.expand_dims(arr, axis=0)
    return arr
