# app/services/classification_service.py
import uuid
from app.utils.image_io import load_image_from_file
from app.ml.classification.predict import predict_leaf_class


def classify_uploaded_image(file_storage):
    """
    Dipanggil oleh endpoint:
    - preprocess image
    - panggil model klasifikasi
    - kembalikan dict siap di-JSON-kan
    """
    # Preprocess citra
    batch = load_image_from_file(file_storage)

    # Prediksi kelas
    label, conf, probs = predict_leaf_class(batch)

    # Sekarang belum pakai DB, jadi ID random saja (UUID)
    analysis_id = str(uuid.uuid4())

    return {
        "id": analysis_id,
        "disease_label": label,
        "confidence": conf,
        "probs": probs,
    }