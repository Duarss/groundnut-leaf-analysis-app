# app/services/classification_service.py
import uuid
from app.utils.temp_store import write_temp_image, write_meta
from app.utils.image_io import load_image_for_classification
from app.ml.classification.predict import predict_leaf_class

_ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "heic"}

def classify_uploaded_image(file_storage):
    img_bytes = file_storage.read()
    analysis_id = str(uuid.uuid4())

    ext = "jpg"
    if getattr(file_storage, "filename", None) and "." in file_storage.filename:
        ext_guess = file_storage.filename.rsplit(".", 1)[-1].lower().strip()
        if ext_guess in _ALLOWED_EXT:
            ext = ext_guess

    write_temp_image(analysis_id, img_bytes, ext=ext)

    x = load_image_for_classification(img_bytes)
    label, conf, probs = predict_leaf_class(x)

    write_meta(analysis_id, {
        "stage": "classified",
        "classification": {
            "label": label,
            "confidence": conf,
            "probs": probs
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
