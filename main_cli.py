# Imports
from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
import numpy as np
from sort import *
from IPython.display import display, clear_output
from PIL import Image
import time

# Initialize YOLO
def initialize_model():
    try:
        model = YOLO("yolov8l.pt")
        print("Loaded yolov8l.pt model")
    except Exception as e:
        print(f"Could not load yolov8l.pt: {e}")
        print("Loading yolov8n.pt instead...")
        model = YOLO("yolov8n.pt")
    return model

# Load video and mask
def load_video_and_mask():
    vid = cv.VideoCapture("/content/assets/traffic_cam.mp4")
    if not vid.isOpened():
        print("Error: Could not open video file '/content/assets/traffic_cam.mp4'")
        return None, None
    mask = cv.imread("/content/assets/mask.png")
    if mask is None:
        print("Warning: Could not load mask file, using full frame instead")
    return vid, mask

# Video writer
def initialize_video_writer(vid):
    width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv.CAP_PROP_FPS)
    if width <= 0 or height <= 0 or fps <= 0:
        print("Error: Could not retrieve video properties")
        vid.release()
        return None
    return cv.VideoWriter("/content/result.mp4", cv.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Graphics
def load_graphic_assets():
    g0 = cv.imread("/content/assets/graphics.png", cv.IMREAD_UNCHANGED)
    g1 = cv.imread("/content/assets/graphics1.png", cv.IMREAD_UNCHANGED)
    if g0 is None: print("Warning: Missing graphics.png")
    if g1 is None: print("Warning: Missing graphics1.png")
    return g0, g1

# Detection
def detect_vehicles(model, frame, class_names):
    result = model(frame, stream=True)
    detections = np.empty((0, 5))
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = class_names[cls]
            if label in ["car", "truck", "bus", "motorbike"]:
                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))
    return detections

# Counting zones
def check_counting_zones(cx, cy, vid_id, count_up, count_down, total_count, line_up, line_down):
    if line_up[0] < cx < line_up[2] and line_up[1] - 5 < cy < line_up[3] + 5:
        if vid_id not in total_count:
            total_count.append(vid_id)
            if vid_id not in count_up:
                count_up.append(vid_id)
    if line_down[0] < cx < line_down[2] and line_down[1] + 15 < cy < line_down[3] + 15:
        if vid_id not in total_count:
            total_count.append(vid_id)
            if vid_id not in count_down:
                count_down.append(vid_id)
    return count_up, count_down, total_count

# Main
model = initialize_model()
vid, mask = load_video_and_mask()
if vid is None:
    raise SystemExit

writer = initialize_video_writer(vid)
if writer is None:
    raise SystemExit

# Class list
class_names = model.names if hasattr(model, "names") else []

# Tracker + lines
tracker = Sort(max_age=22, min_hits=3, iou_threshold=0.3)
line_up = [180, 410, 640, 410]
line_down = [680, 400, 1280, 450]
count_up, count_down, total_count = [], [], []
g0, g1 = load_graphic_assets()

frame_num = 0
display_interval = 20  # display every N frames

while True:
    ret, frame = vid.read()
    if not ret:
        print("End of video.")
        break

    frame_num += 1
    region = cv.bitwise_and(frame, mask) if mask is not None else frame
    detections = detect_vehicles(model, region, class_names)

    if g0 is not None:
        frame = cvzone.overlayPNG(frame, g0, (0, 0))
    if g1 is not None:
        frame = cvzone.overlayPNG(frame, g1, (420, 0))

    tracks = tracker.update(detections)
    cv.line(frame, (line_up[0], line_up[1]), (line_up[2], line_up[3]), (0, 0, 255), 3)
    cv.line(frame, (line_down[0], line_down[1]), (line_down[2], line_down[3]), (0, 0, 255), 3)

    for t in tracks:
        x1, y1, x2, y2, vid_id = map(int, t)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)
        count_up, count_down, total_count = check_counting_zones(cx, cy, vid_id, count_up, count_down, total_count, line_up, line_down)
        cvzone.cornerRect(frame, (x1, y1, w, h), l=5, colorR=(255, 0, 255), rt=1)
        cvzone.putTextRect(frame, f"{vid_id}", (x1, y1), scale=1, thickness=2)

    cv.putText(frame, str(len(total_count)), (255, 100), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), 7)
    cv.putText(frame, str(len(count_up)), (600, 85), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), 7)
    cv.putText(frame, str(len(count_down)), (850, 85), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), 7)

    writer.write(frame)

    if frame_num % display_interval == 0:
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        clear_output(wait=True)
        display(Image.fromarray(rgb))

vid.release()
writer.release()
print("Processing complete. Output saved to /content/result.mp4")
