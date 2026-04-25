"""
Project-wide configuration constants.

This module stores file paths, webcam settings, MediaPipe settings,
and drawing sty;es used by the real-time facial expression demo.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
FACE_DETECTOR_MODEL_PATH = MODELS_DIR / "blaze_face_short_range.tflite"

CAMERA_INDEX = 0

WINDOW_NAME = "DeepFace FERv1"

MIN_FACE_DETECTION_CONFIDENCE = 0.5

BOX_COLOR = (0, 255 ,0)
TEXT_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
TEXT_THICKNESS = 2
FONT_SCALE = 0.7