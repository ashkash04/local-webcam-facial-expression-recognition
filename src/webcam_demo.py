"""Real-time webcam FER demo.

This script captures webcam frames, detects faces with MediaPipe,
classifies each cropped face with DeepFace, and displays the predicted
emotion label on the video feed.
"""

import cv2

from config import (
    BOX_COLOR,
    BOX_THICKNESS,
    CAMERA_INDEX,
    FACE_DETECTOR_MODEL_PATH,
    FONT_SCALE,
    TEXT_COLOR,
    TEXT_THICKNESS,
    WINDOW_NAME,
)

from expression_model import predict_expression
from face_detection import create_face_detector, detect_faces


def draw_prediction(
        frame,
        bbox: tuple[int, int, int, int],
        emotion: str,
        confidence: float,
) -> None:
    """Draw a face bounding box and emotion label on the frame."""
    x1, y1, x2, y2 = bbox

    label = f"{emotion}: {confidence:.2f}"

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        BOX_COLOR,
        BOX_THICKNESS
    )

    cv2.putText(
        frame,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        TEXT_COLOR,
        TEXT_THICKNESS,
        cv2.LINE_AA,
    )


def main() -> None:
    """Run the real-time webcam facial expression recognition demo."""
    if not FACE_DETECTOR_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Face detector model not found: {FACE_DETECTOR_MODEL_PATH}"
        )
    
    face_detector = create_face_detector()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    
    print("Press 'q' to quit.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Failed to read frame from webcam.")
                break

            faces = detect_faces(frame, face_detector)

            for face in faces:
                emotion, confidence = predict_expression(face.crop)

                draw_prediction(
                    frame,
                    face.bbox,
                    emotion,
                    confidence,
                )
            
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()