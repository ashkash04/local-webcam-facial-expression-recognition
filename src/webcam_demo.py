"""Real-time webcam face detection demo using MediaPipe FaceDetector."""

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "blaze_face_short_range.tflite"
WINDOW_NAME = "Face Detection Demo"
CAMERA_INDEX = 0
MIN_DETECTION_CONFIDENCE = 0.5
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
TEXT_FONT = cv2.FONT_HERSHEY_COMPLEX
TEXT_SCALE = 0.7
TEXT_THICKNESS = 2

BaseOptions = mp.tasks.BaseOptions
FaceDetector = vision.FaceDetector
FaceDetectorOptions = vision.FaceDetectorOptions
VisionRunningMode = vision.RunningMode

def create_detector(model_path: Path) -> vision.FaceDetector:
    """Create and return a MediaPipe face detector."""
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )
    return FaceDetector.create_from_options(options)

def draw_detection(frame, detection: mp.tasks.components.containers.Detection) -> None:
    """Draw a bounding box and confidence label for one detected face."""
    bbox = detection.bounding_box
    start_point = (bbox.origin_x, bbox.origin_y)
    end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)

    cv2.rectangle(frame, start_point, end_point, BOX_COLOR, BOX_THICKNESS)

    if detection.categories:
        score = detection.categories[0].score
        label = f"face: {score:.2f}"
        text_point = (bbox.origin_x, max(30, bbox.origin_y - 10))

        cv2.putText(
            frame,
            label,
            text_point,
            TEXT_FONT,
            TEXT_SCALE,
            BOX_COLOR,
            TEXT_THICKNESS
        )

def validate_model_path(model_path: Path) -> bool:
    """Return True if the model file exists, otherwise print an error."""
    if model_path.exists():
        return True
    
    print(f"Error: Model file not found at {model_path}.")
    print(
        "Download the MediaPipe BlazeFace (short-range) model "
        "and save it as models/blaze_face_short_range.tflite."
    )
    return False

def open_webcam(camera_index: int) -> cv2.VideoCapture | None:
    """Open the webcam and return the capture object if successful."""
    cap = cv2.VideoCapture(camera_index)

    if cap.isOpened():
        return cap
    
    print("Error: Could not open webcam.")
    return None

def clamp_bbox(
        x: int,
        y: int,
        w: int,
        h: int,
        frame_width: int,
        frame_height: int,
) -> tuple[int, int, int, int]:
    """Clamp a bounding box to valid frame coordinates."""
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_width, x + w)
    y2 = min(frame_height, y + h)
    return x1, y1, x2, y2

def extract_face_crop(frame, detection):
    """Extract and return a cropped face region for one detection."""
    frame_height, frame_width = frame.shape[:2]
    bbox = detection.bounding_box

    x1, y1, x2, y2 = clamp_bbox(
        bbox.origin_x,
        bbox.origin_y,
        bbox.width,
        bbox.height,
        frame_width,
        frame_height,
    )

    if x2 <= x1 or y2 <= y1:
        return None
    
    return frame[y1:y2, x1:x2]

def predict_expression(face_crop) -> tuple[str, float]:
    """Return a placeholder facial expression prediction."""
    return "neutral", 1.0

def draw_expression_label(frame, detection, expression: str, score: float) -> None:
    """Draw the predicted facial expression above a detected face."""
    bbox = detection.bounding_box
    text = f"{expression}: {score:.2f}"
    text_point = (bbox.origin_x, max(20, bbox.origin_y - 35))

    cv2.putText(
        frame,
        text,
        text_point,
        TEXT_FONT,
        TEXT_SCALE,
        BOX_COLOR,
        TEXT_THICKNESS,
    )

def main() -> None:
    """Run live face detection from the default webcam."""
    if not validate_model_path(MODEL_PATH):
        return
    
    cap = open_webcam(CAMERA_INDEX)
    if cap is None:
        return
    
    start_time = time.monotonic()

    try:
        with create_detector(MODEL_PATH) as detector:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break

                # Flip horizontally so the webcam behaves like a mirror.
                frame = cv2.flip(frame, 1)

                # MediaPipe expects RGB input, but OpenCV captures in BGR.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = int((time.monotonic() - start_time) * 1000)
                result = detector.detect_for_video(mp_image, timestamp_ms)

                for detection in result.detections:
                    draw_detection(frame, detection)

                    face_crop = extract_face_crop(frame, detection)
                    if face_crop is not None and face_crop.size > 0:
                        cv2.imshow("Face Crop", face_crop)
                    
                    if face_crop is None or face_crop.size == 0:
                        continue

                    face_resized = cv2.resize(face_crop, (48, 48))
                    expression, score = predict_expression(face_resized)
                    draw_expression_label(frame, detection, expression, score)
                
                cv2.imshow(WINDOW_NAME, frame)

                # Quit when the user presses q.
                if cv2.waitKey(1) == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()