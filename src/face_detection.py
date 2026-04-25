"""Face detection utilities using MediaPipe.

This module creates a MediaPipe face detector and converts raw detection
results into cropped face images that can be passed to the expression model.
"""

from dataclasses import dataclass

import mediapipe as mp
import numpy as np

from config import FACE_DETECTOR_MODEL_PATH, MIN_FACE_DETECTION_CONFIDENCE
from preprocessing import bgr_to_rgb


@dataclass
class FaceDetectionResult:
    """Stores information for one detected face.

    Attributes:
        bbox: Bounding box coordinates as (x1, y1, x2, y2).
        crop: Cropped face image in OpenCV BGR format.
        confidence: MediaPipe detection confidence score.
    """
    bbox: tuple[int, int, int, int]
    crop: np.ndarray
    confidence: float


def create_face_detector():
    """Create and return a MediaPipe face detector.
    
    Returns:
        A configured MediaPipe FaceDetector instance.
    """
    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(FACE_DETECTOR_MODEL_PATH)
    )

    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
    )

    return mp.tasks.vision.FaceDetector.create_from_options(options)


def detect_faces(frame_bgr: np.ndarray, detector) -> list[FaceDetectionResult]:
    """Detect faces in a webcam frame.
    
    Args:
        frame_bgrL Webcam frame in OpenCV BGR format.
        detector: MediaPipe FaceDetector instance.
        
    Returns:
        A list of detected faces, each containing a bounding box, face crop,
        and detection confidence score.
    """
    frame_rgb = bgr_to_rgb(frame_bgr)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    detection_result = detector.detect(mp_image)

    faces: list[FaceDetectionResult] = []
    height, width = frame_bgr.shape[:2]

    for detection in detection_result.detections:
        bbox = detection.bounding_box

        x1 = max(0, bbox.origin_x)
        y1 = max(0, bbox.origin_y)
        x2 = min(width, bbox.origin_x + bbox.width)
        y2 = min(height, bbox.origin_y + bbox.height)

        if x2 <= x1 or y2 <= y1:
            continue

        face_crop = frame_bgr[y1:y2, x1:x2]

        confidence = 0.0
        if detection.categories:
            confidence = float(detection.categories[0].score)
        
        faces.append(
            FaceDetectionResult(
                bbox=(x1, y1, x2, y2),
                crop=face_crop,
                confidence=confidence,
            )
        )
    
    return faces