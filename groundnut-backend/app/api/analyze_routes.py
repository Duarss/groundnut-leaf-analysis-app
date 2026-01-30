# app/api/analyze_routes.py
from flask import Blueprint, request, jsonify
from app.services.classification_service import classify_uploaded_image
from app.services.segmentation_service import segment_infected_areas
from app.services.save_service import save_analysis
from app.core.config import Config

bp = Blueprint("analyze", __name__)

@bp.route("/classify", methods=["POST"])
def classify():
    """
    Endpoint utama untuk klasifikasi:
    - menerima file "image" (multipart/form-data)
    - mengembalikan JSON berisi label, confidence, dan probabilitas tiap kelas
    """
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada gambar yang terbaca"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Nama file tidak tersedia"}), 400

    try:
        result = classify_uploaded_image(file)
        return jsonify(result), 200
    except Exception as e:
        # Untuk debug sementara, kirim detail error (nanti bisa dihilangkan di production)
        return jsonify({"error": "Proses klasifikasi gagal!", "detail": str(e)}), 500
    
@bp.route("/segment", methods=["POST"])
def segment():
    data = request.get_json(force=True) or {}
    analysis_id = data.get("analysis_id")
    if not analysis_id:
        return jsonify({"error": "analysis_id harus disertakan"}), 400

    try:
        result = segment_infected_areas(analysis_id)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": "Proses segmentasi gagal!", "detail": str(e)}), 500
    
@bp.route("/save", methods=["POST"])
def save():
    data = request.get_json(silent=True) or {}
    analysis_id = data.get("analysis_id")

    if not analysis_id:
        return jsonify({"saved": False, "error": "analysis_id wajib"}), 400
    
    try:
        res = save_analysis(
            analysis_id=analysis_id,
            delete_temp_after=Config.TEMP_DELETE_AFTER_SEG
        )
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"saved": False, "error": str(e)}), 500

