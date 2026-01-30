# app/services/segmentation_service.py
from app.utils.temp_store import (
    read_meta, read_image_bytes, write_meta, is_expired, delete_bundle
)
from app.utils.image_io import load_image_for_segmentation
from app.ml.segmentation.predict import predict_infected_areas

def segment_infected_areas(analysis_id: str):
    meta = read_meta(analysis_id)

    if not meta:
        raise ValueError("Metadata tidak ditemukan. Silahkan lakukan klasifikasi ulang.")

    if is_expired(meta):
        delete_bundle(analysis_id)
        raise ValueError("Mohon upload gambar ulang: data sudah kadaluarsa.")

    cls = (meta.get("classification") or {})
    label = cls.get("label")

    if not label:
        # metadata ada, tapi label kosong → klasifikasi belum tersimpan/format beda
        raise ValueError("Label klasifikasi tidak ditemukan. Silahkan lakukan klasifikasi ulang.")

    # normalisasi agar cocok dengan CLASS_TO_INDEX (yang uppercase semua)
    label_norm = str(label).strip().upper()

    img_bytes = read_image_bytes(analysis_id)
    seg_batch = load_image_for_segmentation(img_bytes)

    seg_result = predict_infected_areas(seg_batch, label_norm)

    # simpan ke meta kalau enabled
    meta["stage"] = "segmented" if seg_result.get("enabled") else meta.get("stage", "classified")
    meta["segmentation"] = seg_result
    write_meta(analysis_id, meta)

    return seg_result
