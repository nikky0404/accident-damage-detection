"""
Backend configuration for 360° Auto Vision.
"""

import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data", "sample_images")
TRAINED_MODELS_DIR = os.path.join(BASE_DIR, "trained_models")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "auto-vision-360-secret-key-2026")
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'accident_vision.db')}")
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB max per request
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    PORT = int(os.getenv("PORT", 5000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
