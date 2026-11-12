"""
Modular AI Routes: Preprocess, Detection, Segmentation, Severity, Fusion, Cost Estimation.
"""

import os
from flask import Blueprint, request, jsonify
from backend.config import UPLOAD_DIR, SAMPLE_DIR, BASE_DIR
from backend.services.inspection_service import InspectionService

ai_bp = Blueprint("ai", __name__)
service = InspectionService()

def _resolve_image_path(path_str):
    if not path_str:
        return None
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str
    if os.path.exists(os.path.join(BASE_DIR, path_str)):
        return os.path.join(BASE_DIR, path_str)
    c1 = os.path.join(UPLOAD_DIR, os.path.basename(path_str))
    if os.path.exists(c1):
        return c1
    c2 = os.path.join(SAMPLE_DIR, os.path.basename(path_str))
    if os.path.exists(c2):
        return c2
    return None

@ai_bp.route("/preprocess", methods=["POST"])
def preprocess_endpoint():
    """POST /api/preprocess"""
    data = request.json or {}
    img_path = _resolve_image_path(data.get("image_path"))
    if not img_path:
        return jsonify({"success": False, "error": "Image file not found"}), 400

    try:
        res = service.preprocess_image(img_path)
        return jsonify({"success": True, "result": res}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@ai_bp.route("/detect", methods=["POST"])
def detect_parts_endpoint():
    """POST /api/detect (Vehicle & Body-Part Detection)"""
    data = request.json or {}
    img_path = _resolve_image_path(data.get("image_path"))
    view_name = data.get("view_name", "Front View")
    if not img_path:
        return jsonify({"success": False, "error": "Image file not found"}), 400

    try:
        annotated_img, detections = service.part_detector.detect_parts(img_path, view_name)
        out_name = f"detected_parts_{os.path.basename(img_path)}"
        out_path = os.path.join(UPLOAD_DIR, out_name)
        import cv2
        cv2.imwrite(out_path, annotated_img)
        return jsonify({
            "success": True,
            "annotated_image": out_name,
            "detections": detections
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@ai_bp.route("/damage-detect", methods=["POST"])
def damage_detect_endpoint():
    """POST /api/damage-detect (Damage Detection)"""
    data = request.json or {}
    img_path = _resolve_image_path(data.get("image_path"))
    parts = data.get("parts", [])
    view_name = data.get("view_name", "Front View")
    if not img_path:
        return jsonify({"success": False, "error": "Image file not found"}), 400

    try:
        annotated_img, damages = service.damage_detector.detect_damage(img_path, parts, view_name)
        out_name = f"damage_detected_{os.path.basename(img_path)}"
        out_path = os.path.join(UPLOAD_DIR, out_name)
        import cv2
        cv2.imwrite(out_path, annotated_img)
        return jsonify({
            "success": True,
            "annotated_image": out_name,
            "damages": damages
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@ai_bp.route("/segment", methods=["POST"])
def segment_endpoint():
    """POST /api/segment (Damage Segmentation)"""
    data = request.json or {}
    img_path = _resolve_image_path(data.get("image_path"))
    damages = data.get("damages", [])
    if not img_path:
        return jsonify({"success": False, "error": "Image file not found"}), 400

    try:
        seg_img, binary_mask, seg_damages = service.segmenter.segment_damage(img_path, damages)
        seg_name = f"seg_{os.path.basename(img_path)}"
        mask_name = f"mask_{os.path.basename(img_path)}"
        import cv2
        cv2.imwrite(os.path.join(UPLOAD_DIR, seg_name), seg_img)
        cv2.imwrite(os.path.join(UPLOAD_DIR, mask_name), binary_mask)
        return jsonify({
            "success": True,
            "segmented_image": seg_name,
            "mask_image": mask_name,
            "damages": seg_damages
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@ai_bp.route("/severity", methods=["POST"])
def severity_endpoint():
    """POST /api/severity (Damage Severity Classification)"""
    data = request.json or {}
    damages = data.get("damages", [])
    try:
        classified = service.severity_classifier.batch_classify(damages)
        return jsonify({"success": True, "damages": classified}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@ai_bp.route("/fuse", methods=["POST"])
def fuse_endpoint():
    """POST /api/fuse (360 Multi-Image Fusion)"""
    data = request.json or {}
    view_results = data.get("view_results", [])
    try:
        fused = service.fusion.fuse_inspections(view_results)
        return jsonify({"success": True, "fusion": fused}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@ai_bp.route("/estimate-cost", methods=["POST"])
def estimate_cost_endpoint():
    """POST /api/estimate-cost (AI Repair Cost Estimation)"""
    data = request.json or {}
    vehicle_info = data.get("vehicle_info", {"manufacturer": "Hyundai", "model": "Creta", "year": 2023})
    fused_damages = data.get("damages", [])
    try:
        costs = service.cost_estimator.estimate_total_inspection(vehicle_info, fused_damages)
        return jsonify({"success": True, "cost_estimation": costs}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
