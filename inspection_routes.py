"""
Inspection Routes: Upload, Full Pipeline, Report, History, and Sample Data.
"""

import os
import uuid
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
from backend.config import UPLOAD_DIR, REPORT_DIR, SAMPLE_DIR, BASE_DIR
from backend.services.inspection_service import InspectionService
from database.database import SessionLocal
from database.models import Inspection, DamageRecord, Vehicle

inspection_bp = Blueprint("inspection", __name__)
service = InspectionService()

@inspection_bp.route("/upload", methods=["POST"])
def upload_images():
    """
    POST /api/upload
    Receives multi-part file uploads tagged by view name.
    """
    if "files" not in request.files and len(request.files) == 0:
        return jsonify({"success": False, "error": "No file part in the request"}), 400

    uploaded = []
    # Can send multiple files under 'files' or individual view keys (e.g. 'front_view')
    for key, file in request.files.items():
        if file and file.filename:
            orig_name = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex[:8]}_{orig_name}"
            dest = os.path.join(UPLOAD_DIR, unique_name)
            file.save(dest)
            
            view_label = request.form.get(f"{key}_view", key.replace("_", " ").title())
            uploaded.append({
                "original_name": orig_name,
                "saved_name": unique_name,
                "view_name": view_label,
                "path": dest,
                "size_bytes": os.path.getsize(dest)
            })

    return jsonify({
        "success": True,
        "count": len(uploaded),
        "files": uploaded
    }), 200

@inspection_bp.route("/inspection/full-pipeline", methods=["POST"])
def run_full_pipeline():
    """
    POST /api/inspection/full-pipeline
    Receives either uploaded file paths or sample image selections, plus vehicle metadata.
    """
    try:
        data = request.json or {}
        views_data = data.get("views", [])
        vehicle_info = data.get("vehicle_info", {
            "manufacturer": "Hyundai",
            "model": "Creta SX",
            "year": 2023,
            "registration_number": "MH-12-AB-1234",
            "accident_date": "2026-10-01"
        })

        if not views_data:
            # Fall back to sample images if none provided
            views_data = [
                {"path": os.path.join(SAMPLE_DIR, "Front_View.jpg"), "view_name": "Front View"},
                {"path": os.path.join(SAMPLE_DIR, "Front-Left_View.jpg"), "view_name": "Front-Left View"},
                {"path": os.path.join(SAMPLE_DIR, "Left_View.jpg"), "view_name": "Left View"},
                {"path": os.path.join(SAMPLE_DIR, "Rear_View.jpg"), "view_name": "Rear View"}
            ]

        file_pairs = []
        for v in views_data:
            p = v.get("path", "")
            if os.path.isabs(p) and os.path.exists(p):
                file_pairs.append((p, v.get("view_name", "Exterior View")))
            elif os.path.exists(os.path.join(BASE_DIR, p)):
                file_pairs.append((os.path.join(BASE_DIR, p), v.get("view_name", "Exterior View")))
            elif os.path.exists(os.path.join(UPLOAD_DIR, os.path.basename(p))):
                file_pairs.append((os.path.join(UPLOAD_DIR, os.path.basename(p)), v.get("view_name", "Exterior View")))
            elif os.path.exists(os.path.join(SAMPLE_DIR, os.path.basename(p))):
                file_pairs.append((os.path.join(SAMPLE_DIR, os.path.basename(p)), v.get("view_name", "Exterior View")))

        if not file_pairs:
            return jsonify({"success": False, "error": "No valid image files accessible on server"}), 400

        result = service.run_full_360_pipeline(file_pairs, vehicle_info)
        return jsonify({"success": True, "data": result}), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@inspection_bp.route("/report/<int:inspection_id>", methods=["GET"])
def get_report_json(inspection_id):
    """GET /api/report/<inspection_id>"""
    db = SessionLocal()
    try:
        insp = db.query(Inspection).filter_by(id=inspection_id).first()
        if not insp:
            return jsonify({"success": False, "error": "Inspection record not found"}), 404

        damages = [d.to_dict() for d in insp.damage_records]
        data = insp.to_dict()
        data["damage_records"] = damages
        return jsonify({"success": True, "report": data}), 200
    finally:
        db.close()

@inspection_bp.route("/report/<int:inspection_id>/pdf", methods=["GET"])
def get_report_pdf(inspection_id):
    """GET /api/report/<inspection_id>/pdf"""
    pdf_filename = f"damage_assessment_report_{inspection_id}.pdf"
    pdf_path = os.path.join(REPORT_DIR, pdf_filename)
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True, download_name=pdf_filename, mimetype="application/pdf")
    return jsonify({"success": False, "error": "PDF not generated for this inspection"}), 404

@inspection_bp.route("/history", methods=["GET"])
def get_inspection_history():
    """GET /api/history"""
    db = SessionLocal()
    try:
        inspections = db.query(Inspection).order_by(Inspection.created_at.desc()).limit(50).all()
        history = []
        for insp in inspections:
            item = insp.to_dict()
            item["damage_count"] = len(insp.damage_records)
            history.append(item)
        return jsonify({"success": True, "history": history}), 200
    finally:
        db.close()

@inspection_bp.route("/sample-images", methods=["GET"])
def get_sample_images():
    """GET /api/sample-images"""
    files = []
    if os.path.exists(SAMPLE_DIR):
        for f in os.listdir(SAMPLE_DIR):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                view_name = f.replace(".jpg", "").replace("_", " ")
                files.append({
                    "filename": f,
                    "view_name": view_name,
                    "relative_path": os.path.join("sample_data", "sample_images", f)
                })
    return jsonify({"success": True, "samples": files}), 200
