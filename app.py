"""
360° Auto Vision – Accident Damage Detection & Cost Estimation Backend Server.
Flask REST API application.
"""

import os
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from backend.config import Config, UPLOAD_DIR, REPORT_DIR, SAMPLE_DIR
from database.database import init_db
from backend.routes.inspection_routes import inspection_bp
from backend.routes.ai_routes import ai_bp
from backend.routes.dashboard_routes import dashboard_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Enable CORS for cross-origin Streamlit frontend requests
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Initialize Database tables and master pricing catalogs
    init_db()

    # Register Blueprints
    app.register_blueprint(inspection_bp, url_prefix="/api")
    app.register_blueprint(ai_bp, url_prefix="/api")
    app.register_blueprint(dashboard_bp, url_prefix="/api")

    # Static file routes for uploads, reports, and sample images
    @app.route("/uploads/<path:filename>")
    def serve_upload(filename):
        return send_from_directory(UPLOAD_DIR, filename)

    @app.route("/reports/<path:filename>")
    def serve_report(filename):
        return send_from_directory(REPORT_DIR, filename)

    @app.route("/samples/<path:filename>")
    def serve_sample(filename):
        return send_from_directory(SAMPLE_DIR, filename)

    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({
            "status": "healthy",
            "service": "360° Auto Vision AI Engine",
            "version": "1.0.0",
            "environment": "production-ready",
            "ai_models": {
                "vehicle_detector": "YOLOv8 Active",
                "damage_detector": "YOLOv8/11 Active",
                "damage_segmenter": "YOLO-Seg Active",
                "severity_classifier": "CNN Active",
                "cost_estimator": "XGBoost / Random Forest Active",
                "multi_image_fusion": "ORB Feature Matching Active"
            }
        }), 200

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Resource not found", "status": 404}), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return jsonify({"error": "Internal AI Processing Error", "details": str(e), "status": 500}), 500

    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"================================================================")
    print(f"  360° AUTO VISION - AI AUTOMOTIVE INSPECTION BACKEND API")
    print(f"  Running on: http://127.0.0.1:{port}")
    print(f"================================================================")
    app.run(host="0.0.0.0", port=port, debug=True)
