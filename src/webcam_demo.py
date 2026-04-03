import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "blaze_face_short_range.tflite"

BaseOptions = mp.tasks.BaseOptions
FaceDetector = vision.FaceDetector
FaceDetectorOptions = vision.FaceDetectorOptions
VisionRunningMode = vision.RunningMode

def main() -> None:
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}.")
        print("Download a MediaPipe face detector task model (like BlazeFace) and save it as models/blaze_face_short_range.tflite")
        return
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        min_detection_confidence=0.5
    )

    with FaceDetector.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            if not ret:
                print("Error: Could not read frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            for detection in result.detections:
                bbox = detection.bounding_box
                start_point = (bbox.origin_x, bbox.origin_y)
                end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
                cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

                if detection.categories:
                    score = detection.categories[0].score
                    label = f"face: {score:.2f}"
                    text_point = (bbox.origin_x, max(30, bbox.origin_y - 10))
                    cv2.putText(
                        frame,
                        label,
                        text_point,
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
            
            cv2.imshow("Face Detection Demo", frame)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()