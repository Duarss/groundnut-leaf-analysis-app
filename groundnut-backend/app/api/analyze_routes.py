# app/api/analyze_routes.py
from flask import Blueprint, request, jsonify
from app.services.classification_service import classify_uploaded_image

analyze_bp = Blueprint("analyze", __name__)

@analyze_bp.route("/analyze", methods=["POST"])
def analyze():
    """
    Endpoint utama untuk klasifikasi:
    - menerima file "image" (multipart/form-data)
    - mengembalikan JSON berisi label, confidence, dan probabilitas tiap kelas
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        result = classify_uploaded_image(file)
        return jsonify(result), 200
    except Exception as e:
        # Untuk debug sementara, kirim detail error (nanti bisa dihilangkan di production)
        return jsonify({"error": "Failed to classify image", "detail": str(e)}), 500
