# local-webcam-facial-expression-recognition

Real-time local facial expression recognition from webcam video using OpenCV, MediaPipe, and DeepFace.

This project detects faces from a live webcam feed, crops each detected face, and classifies the dominant facial expression using a pretrained DeepFace emotion model.

## Features

- Real-time webcam video capture with OpenCV
- Face detection using MediaPipe
- Cropped-face emotion classification using DeepFace
- On-screen bounding boxes, emotion labels, and confidence scores
- Fully local webcam processing

## Tech Stack

- Python
- OpenCV
- MediaPipe
- DeepFace
- TensorFlow / Keras
- NumPy

## Requirements

- Python 3.10 recommended
- Webcam
- Windows (tested)

## Quick Start

```powershell
git clone https://github.com/ashkash04/local-webcam-facial-expression-recognition.git
cd local-webcam-facial-expression-recognition

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

python .\src\webcam_demo.py
```

Press 'q' to quit the webcam window.