# app/api/analyze_routes.py
from flask import Blueprint, request, jsonify, send_file
from app.utils.temp_store import find_image_path
from app.services.classification_service import classify_uploaded_image
from app.services.segmentation_service import segment_infected_areas
from app.services.save_service import save_analysis
from app.services.history_service import list_history, get_history_detail
from app.core.config import Config

bp = Blueprint("analyze", __name__)


def _get_client_id():
    cid = request.headers.get("X-Client-Id")
    if cid:
        return str(cid).strip()

    cid = request.args.get("client_id")
    if cid:
        return str(cid).strip()

    return None

@bp.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada gambar yang terbaca"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nama file tidak tersedia"}), 400

    try:
        result = classify_uploaded_image(file)
        return jsonify(result), 200
    except Exception as e:
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

    cid = _get_client_id()
    if not cid:
        return jsonify({"saved": False, "error": "client_id wajib (untuk multi-user tanpa login)."}), 400

    try:
        res = save_analysis(
            analysis_id=analysis_id,
            client_id=cid,
            delete_temp_after=Config.TEMP_DELETE_AFTER_SEG,
        )
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"saved": False, "error": str(e)}), 500


@bp.route("/history", methods=["GET"])
def history_list():
    cid = _get_client_id()
    if not cid:
        return jsonify({"error": "client_id wajib (untuk multi-user tanpa login)."}), 400

    try:
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))
    except Exception:
        return jsonify({"error": "limit/offset harus integer"}), 400

    try:
        data = list_history(client_id=cid, limit=limit, offset=offset)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": "Gagal mengambil history", "detail": str(e)}), 500


@bp.route("/history/<analysis_id>", methods=["GET"])
def history_detail(analysis_id):
    try:
        client_id = _get_client_id()
        detail = get_history_detail(analysis_id, client_id)
        if not detail:
            return jsonify({"error": "History tidak ditemukan"}), 404
        return jsonify(detail), 200
    except Exception as e:
        return jsonify({"error": "Gagal mengambil detail history", "detail": str(e)}), 500


@bp.route("/temp-image/<analysis_id>", methods=["GET"])
def temp_image(analysis_id):
    """
    Serve original image dari tmp_uploads berdasarkan analysis_id.
    Dipakai oleh SegmentPage agar tidak bergantung pada sessionStorage.

    GET /api/temp-image/<analysis_id>
    """
    try:
        p = find_image_path(analysis_id)
        if not p:
            return jsonify({"error": "File gambar temporary tidak ditemukan"}), 404
        return send_file(p, as_attachment=False)
    except Exception as e:
        return jsonify({"error": "Gagal mengambil citra temporary", "detail": str(e)}), 500