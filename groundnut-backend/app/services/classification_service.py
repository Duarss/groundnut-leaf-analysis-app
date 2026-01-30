# app/services/classification_service.py
import uuid
import numpy as np
from app.utils.temp_store import write_temp_image, write_meta
from app.utils.image_io import load_image_for_classification
from app.ml.classification.predict import predict_leaf_class

def classify_uploaded_image(file_storage):
    """
    Memproses file upload, menyimpan sementara, melakukan klasifikasi,
    dan menyimpan metadata klasifikasi.
    """
    img_bytes = file_storage.read()
    analysis_id = str(uuid.uuid4())

    # Simpan sementara (ext optional: baca dari filename)
    ext = "jpg"
    if getattr(file_storage, "filename", None) and "." in file_storage.filename:
        ext = file_storage.filename.rsplit(".", 1)[-1]

    # Simpan sementara agar bisa pakai ulang utk segmentasi    
    write_temp_image(analysis_id, img_bytes, ext=ext)

    # Klasifikasi
    x = load_image_for_classification(img_bytes)
    label, conf, probs = predict_leaf_class(x)

    write_meta(analysis_id, {
        "stage": "classified",
        "classification": {
            "label": label,
            "confidence": conf,
            "probs": probs # Pastikan probs JSON-serializable (list/dict)
        }
    })

    segmentation_ready = (str(label).lower() != "healthy")

    return {
        "analysis_id": analysis_id,
        "label": label,
        "confidence": conf,
        "probs": probs,
        "segmentation_ready": segmentation_ready,
        "message": "Healthy -> segmentasi dinonaktifkan" if not segmentation_ready else "Siap untuk segmentasi"
    }