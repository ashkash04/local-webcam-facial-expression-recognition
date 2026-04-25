"""
Image preprocessing utilities.

This module contains small helper functions for validating images and
converting between OpenCV's BGR format and RGB format used by MediaPipe.
"""

import cv2
import numpy as np


def validate_image(image: np.ndarray) -> None:
    if image is None or image.size == 0:
        raise ValueError("Received an empty image.")
    
    
def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    validate_image(image_bgr)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)