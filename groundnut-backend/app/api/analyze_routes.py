# app/api/analyze_routes.py
from flask import Blueprint, request, jsonify
from app.services.classification_service import classify_uploaded_image
from app.services.segmentation_service import segment_infected_areas
from app.services.save_service import save_analysis
from app.services.history_service import list_history, get_history_detail
from app.core.config import Config

bp = Blueprint("analyze", __name__)

def _get_client_id():
    # prioritas header (sesuai frontend kamu)
    cid = request.headers.get("X-Client-Id")
    if cid:
        return str(cid).strip()

    # fallback: query param
    cid = request.args.get("client_id")
    if cid:
        return str(cid).strip()

    return None

def _to_iso(dt):
    # aman untuk datetime/timestamp maupun string
    try:
        return dt.isoformat()
    except Exception:
        return str(dt) if dt is not None else None


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
    
    cid = _get_client_id()

    if not cid:
        return jsonify({"saved": False, "error": "client_id wajib (untuk multi-user tanpa login)."}), 400
    
    try:
        res = save_analysis(
            analysis_id=analysis_id,
            client_id=cid,
            delete_temp_after=Config.TEMP_DELETE_AFTER_SEG
        )
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"saved": False, "error": str(e)}), 500
    
@bp.route("/history", methods=["GET"])
def history_list():
    """
    GET /api/history?limit=50&offset=0
    Header: X-Client-Id: <id>
    """
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

        if isinstance(data, dict):
            # pastikan created_at string
            items = data.get("items") or []
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict) and "created_at" in it:
                        it["created_at"] = _to_iso(it["created_at"])
            return jsonify(data), 200

        # kalau service mengembalikan list rows ORM / dict
        rows = data if isinstance(data, list) else []
        items = []
        for r in rows:
            if isinstance(r, dict):
                items.append({
                    "analysis_id": r.get("analysis_id"),
                    "created_at": _to_iso(r.get("created_at")),
                    "label": r.get("label"),
                    "confidence": r.get("confidence"),
                    "severity_pct": r.get("severity_pct"),
                })
            else:
                items.append({
                    "analysis_id": getattr(r, "analysis_id", None),
                    "created_at": _to_iso(getattr(r, "created_at", None)),
                    "label": getattr(r, "label", None),
                    "confidence": getattr(r, "confidence", None),
                    "severity_pct": getattr(r, "severity_pct", None),
                })

        return jsonify({"items": items, "limit": limit, "offset": offset}), 200

    except Exception as e:
        return jsonify({"error": "Gagal mengambil history", "detail": str(e)}), 500
    
@bp.route("/history/<analysis_id>", methods=["GET"])
def history_detail(analysis_id):
    try:
        client_id = _get_client_id()  # <-- INI yang sering lupa
        detail = get_history_detail(analysis_id, client_id)
        if not detail:
            return {"error": "History tidak ditemukan"}, 404
        return detail, 200
    except Exception as e:
        return {"error": "Gagal mengambil detail history", "detail": str(e)}, 500
