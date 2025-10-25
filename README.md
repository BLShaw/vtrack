# vtrack - Vehicle Tracking and Counting System

A real-time vehicle detection and counting system using YOLO object detection and SORT tracking algorithm.

## Overview

vtrack is a computer vision application that processes video footage to detect, track, and count vehicles in traffic. It uses state-of-the-art deep learning models to identify vehicles and sophisticated tracking algorithms to maintain unique IDs across frames, allowing accurate counting of vehicles moving in different directions.

## Features

- Real-time vehicle detection using YOLO (You Only Look Once)
- Multi-object tracking with SORT (Simple, Online and Realtime Tracking)
- Direction-based vehicle counting (up/down lanes)
- Visual overlays for tracking visualization
- Video output with bounding boxes, IDs, and counts
- Configurable counting lines and detection parameters

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BLShaw/vtrack.git
   cd vtrack
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your traffic video file in the `assets/` directory
2. Update the video path in `main.py` if needed (default is `assets/traffic_cam.mp4`)
3. Configure counting line positions in `main.py` based on your video
4. Run the application:
   ```bash
   python main.py
   ```

The system will:
- Detect vehicles (cars, trucks, buses, motorcycles)
- Track them across frames using unique IDs
- Count vehicles crossing defined lines
- Display real-time counts on screen
- Save the processed video as `result.mp4`

## Configuration

### Counting Lines
You can adjust the counting lines in `main.py`:
- `line_up`: Line for vehicles going in the upward direction
- `line_down`: Line for vehicles going in the downward direction

The format is `[x1, y1, x2, y2]` representing coordinates of the line.

### Detection Parameters
- `tracker`: SORT tracker parameters (max_age, min_hits, iou_threshold)
- Vehicle classes: Currently configured for "car", "truck", "bus", "motorbike"

## File Structure

```
vtrack/
├── main.py            # Main application logic
├── main_cli.py        # For CLI enviornments
├── sort.py            # SORT tracking algorithm
├── requirements.txt   # Python dependencies
├── README.md         # This file
└── assets/
    ├── traffic_cam.mp4  # Input video
    ├── mask.png         # Region of interest mask
    ├── graphics.png     # UI overlay graphics
    └── graphics1.png    # UI overlay graphics
```

## Customization

- **Adjust Detection Area**: Modify `mask.png` to limit detection to specific areas
- **Change Vehicle Types**: Modify the vehicle class filter in `main.py`
- **Update Counting Lines**: Adjust `line_up` and `line_down` coordinates
- **Change Model**: Update the YOLO model in `main.py` (e.g., yolov8n.pt, yolov8m.pt)

## Acknowledgments

- YOLO for the object detection model
- SORT algorithm for multi-object tracking
- OpenCV for computer vision operations
- cvzone for computer vision utilities