"""DeepFace expression recognition wrapper.

This module provides a small project-level function for predicting
facial expressions from cropped face images using DeepFace.
"""

import numpy as np
from deepface import DeepFace


def predict_expression(face_crop_bgr: np.ndarray) -> tuple[str, float]:
    """
    Predict the dominant emotion from a cropped face image.

    Args:
        face_crop_bgr: Cropped face image in OpenCV BGR format.

    Returns:
        A tuple containing the predicted emotion label and confidence score.
        The confidence score is normalized from 0.0 to 1.0.
    """
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return "unknown", 0.0
    
    try:
        result = DeepFace.analyze(
            img_path=face_crop_bgr,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="skip",
            silent=True,
        )

        if isinstance(result, list):
            result = result[0]
        
        emotion = result.get("dominant_emotion", "unknown")
        emotion_scores = result.get("emotion", {})

        raw_confidence = emotion_scores.get(emotion, 0.0)
        confidence = float(raw_confidence) / 100.0

        return emotion, confidence

    except Exception as error:
        print(f"Expression prediction failed: {error}")
        return "unknown", 0.0