from ultralytics import YOLO
import numpy as np
import math
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def initialize_model(model_name: str, fallback_name: str) -> YOLO:
    """Initialize the YOLO model with fallback options."""
    try:
        model = YOLO(model_name)
        logger.info(f"Loaded {model_name} model")
    except Exception as e:
        logger.warning(f"Could not load {model_name}: {e}")
        logger.info(f"Loading {fallback_name} instead...")
        model = YOLO(fallback_name)
    return model

def detect_vehicles(
    model: YOLO, 
    frame: np.ndarray, 
    class_names: Dict[int, str], 
    target_classes: List[str]
) -> np.ndarray:
    """Detect vehicles in a frame using the YOLO model."""
    result = model(frame, stream=True)
    detections = np.empty((0, 5))
    
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = (x2-x1), (y2-y1)

            conf = math.floor(box.conf[0]*100)/100

            cls = int(box.cls[0])
            vehicle_names = class_names.get(cls, "unknown")

            if vehicle_names in target_classes:
                current_detection = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_detection))
    
    return detections