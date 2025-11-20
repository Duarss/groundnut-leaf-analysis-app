# app/utils/image_io.py
import io
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from app.core.config import Config


def load_image_from_file(file_storage):
    """
    Konversi Flask FileStorage -> batch numpy (1, H, W, 3) yang sudah di-preprocess
    untuk EfficientNet-B4.
    """
    # Baca bytes dari upload
    image_bytes = file_storage.read()

    # Buka dengan PIL, pastikan RGB
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize ke ukuran input EfficientNet-B4
    img = img.resize((Config.IMG_WIDTH, Config.IMG_HEIGHT))

    # Ke numpy array float32
    arr = np.array(img, dtype=np.float32)

    # Tambah dimensi batch
    arr = np.expand_dims(arr, axis=0)

    # Preprocess sesuai EfficientNet (normalisasi dsb.)
    arr = preprocess_input(arr)

    return arr
