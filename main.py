from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
import numpy as np
from sort import *

def initialize_model():
    """Initialize the YOLO model with fallback options."""
    try:
        model = YOLO("yolov8l.pt")  # This will download the model if not found locally
        print("Loaded yolov8l.pt model")
    except Exception as e:
        print(f"Could not load yolov8l.pt: {e}")
        print("Loading yolov8n.pt instead...")
        model = YOLO("yolov8n.pt")  # Fallback to a different model size
    return model

def load_video_and_mask():
    """Load video and mask files with error checking."""
    # Load video and check if it opened successfully
    vid = cv.VideoCapture("assets/traffic_cam.mp4")
    if not vid.isOpened():
        print("Error: Could not open video file 'assets/traffic_cam.mp4'")
        return None, None
        
    # Load mask and check if it loaded successfully
    mask = cv.imread("assets/mask.png")  # For blocking out noise
    if mask is None:
        print("Warning: Could not load mask file 'assets/mask.png', using original frame instead")
        
    return vid, mask

def initialize_video_writer(vid):
    """Initialize the video writer with error checking."""
    width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv.CAP_PROP_FPS)

    # Check if video properties were retrieved successfully
    if width <= 0 or height <= 0 or fps <= 0:
        print("Error: Could not retrieve video properties")
        vid.release()
        return None

    video_writer = cv.VideoWriter(("result.mp4"), cv.VideoWriter_fourcc("m", "p", "4", "v"),
                                  fps, (width, height))
    return video_writer

def load_graphic_assets():
    """Load graphic overlay assets with error checking."""
    # Total count graphics
    frame_graphics = cv.imread("assets/graphics.png", cv.IMREAD_UNCHANGED)
    if frame_graphics is None:
        print("Warning: Could not load graphics.png")
        
    # Vehicle count graphics
    frame_graphics1 = cv.imread("assets/graphics1.png", cv.IMREAD_UNCHANGED)
    if frame_graphics1 is None:
        print("Warning: Could not load graphics1.png")
    
    return frame_graphics, frame_graphics1

def detect_vehicles(model, frame, class_names):
    """Detect vehicles in a frame using the YOLO model."""
    result = model(frame, stream=True)
    detections = np.empty((0, 5))

    for r in result:
        boxes = r.boxes
        for box in boxes:
            # Bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = (x2-x1), (y2-y1)

            # Detection confidence
            conf = math.floor(box.conf[0]*100)/100

            # Class names
            cls = int(box.cls[0])
            vehicle_names = class_names[cls]

            if vehicle_names in ["car", "truck", "bus", "motorbike"]:
                current_detection = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_detection))
    
    return detections

def check_counting_zones(cx, cy, id, count_up, count_down, total_count, line_up, line_down):
    """Check if a vehicle has crossed any of the counting lines."""
    # Code for left lane (vehicles driving up)
    if line_up[0] < cx < line_up[2] and line_up[1] - 5 < cy < line_up[3] + 5:
        if total_count.count(id) == 0:
            total_count.append(id)
            if count_up.count(id) == 0:
                count_up.append(id)

    # Code for right lane (vehicles driving down)
    if line_down[0] < cx < line_down[2] and line_down[1] + 15 < cy < line_down[3] + 15:
        if total_count.count(id) == 0:
            total_count.append(id)
            if count_down.count(id) == 0:
                count_down.append(id)

    return count_up, count_down, total_count

# Initialize model
model = initialize_model()

# Initialize video, mask, and video writer
vid, mask = load_video_and_mask()
if vid is None:
    exit()

video_writer = initialize_video_writer(vid)
if video_writer is None:
    vid.release()
    exit()

# Load class names for YOLO model
class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Configuration parameters
tracker = Sort(max_age = 22, min_hits = 3, iou_threshold = 0.3)

# Counting line coordinates - can be adjusted based on the specific video
# Format: [x1, y1, x2, y2] representing the line from (x1, y1) to (x2, y2)
line_up = [180, 410, 640, 410]      # Line for vehicles going up
line_down = [680, 400, 1280, 450]   # Line for vehicles going down

# Vehicle count tracking
count_up = []      # Count of vehicles going up
count_down = []    # Count of vehicles going down
total_count = []   # Total unique vehicles counted

# Load graphic assets
frame_graphics, frame_graphics1 = load_graphic_assets()

while True:
    ret, frame = vid.read()
    if not ret:
        print("End of video or error reading frame")
        break
    
    # Apply mask if available
    if mask is not None:
        frame_region = cv.bitwise_and(frame, mask)
    else:
        frame_region = frame

    # Detect vehicles in the frame
    detections = detect_vehicles(model, frame_region, class_names)

    # Total count graphics
    if frame_graphics is not None:
        frame = cvzone.overlayPNG(frame, frame_graphics, (0,0))

    # Vehicle count graphics
    if frame_graphics1 is not None:
        frame = cvzone.overlayPNG(frame, frame_graphics1, (420,0))

    # Tracking codes
    tracker_updates = tracker.update(detections)
    # Tracking lines
    cv.line(frame, (line_up[0], line_up[1]), (line_up[2], line_up[3]), (0, 0, 255), thickness = 3)
    cv.line(frame, (line_down[0] ,line_down[1]), (line_down[2], line_down[3]), (0, 0, 255), thickness = 3)

    # Getting bounding boxes points and vehicle ID
    for update in tracker_updates:
        x1, y1, x2, y2, id = update
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = (x2-x1), (y2-y1)

        # Getting tracking marker
        cx, cy = (x1+w//2), (y1+h//2)
        cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)

        # Check counting zones for this vehicle
        count_up, count_down, total_count = check_counting_zones(cx, cy, id, count_up, count_down, total_count, line_up, line_down)

        # Adding rectangles and texts
        cvzone.cornerRect(frame, (x1, y1, w, h), l=5, colorR=(255, 0, 255), rt=1)
        cvzone.putTextRect(frame, f'{id}', (x1, y1), scale=1, thickness=2)

    # Adding texts to graphics
    cv.putText(frame, str(len(total_count)), (255, 100), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), thickness=7)
    cv.putText(frame, str(len(count_up)), (600, 85), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), thickness=7)
    cv.putText(frame, str(len(count_down)), (850, 85), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), thickness=7)

    cv.imshow("vid", frame)

    # Saving the video frame output
    video_writer.write(frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Closing down everything
vid.release()
cv.destroyAllWindows()
video_writer.release()