# app/services/segmentation_service.py
from app.utils.temp_store import read_meta, read_image_bytes, write_meta, is_expired, delete_bundle
from app.utils.image_io import load_image_for_segmentation
from app.ml.segmentation.predict import predict_infected_areas
from app.core.config import Config

def segment_by_analysis_id(analysis_id: str):
    meta = read_meta(analysis_id)
    if is_expired(meta):
        delete_bundle(analysis_id)
        raise ValueError("Mohon upload gambar ulang: data sudah kadaluarsa.")
    
    cls = meta.get("classification", {})
    label = cls.get("label")
    if not label:
        raise ValueError("Label klasifikasi tidak ditemukan. Silahkan lakukan klasifikasi ulang.")
    
    img_bytes = read_image_bytes(analysis_id)
    seg_batch = load_image_for_segmentation(img_bytes)

    seg_result = predict_infected_areas(seg_batch, label)

    # Simpan hasil segmentasi ke meta
    meta["stage"] = "segmented"
    meta["segmentation"] = seg_result
    write_meta(analysis_id, meta)

    return seg_result