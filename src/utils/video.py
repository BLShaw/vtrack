import cv2 as cv
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def load_video_and_mask(
    video_path: str,
    mask_path: str
) -> Tuple[Optional[cv.VideoCapture], Optional[np.ndarray]]:
    """Load video and mask files with error checking."""
    
    vid = cv.VideoCapture(video_path)
    if not vid.isOpened():
        logger.error(f"Could not open video file '{video_path}'")
        return None, None
        
    mask = cv.imread(mask_path)
    if mask is None:
        logger.warning(f"Could not load mask file '{mask_path}', using original frame instead")
        
    return vid, mask

def initialize_video_writer(
    vid: cv.VideoCapture, 
    output_path: str
) -> Optional[cv.VideoWriter]:
    """Initialize the video writer with error checking."""
    width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv.CAP_PROP_FPS)

    if width <= 0 or height <= 0 or fps <= 0:
        logger.error("Could not retrieve video properties")
        vid.release()
        return None

    video_writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc("m", "p", "4", "v"),
                                  fps, (width, height))
    return video_writer

def load_graphic_assets(
    graphics_path: str,
    graphics1_path: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load graphic overlay assets with error checking."""
    
    frame_graphics = cv.imread(graphics_path, cv.IMREAD_UNCHANGED)
    if frame_graphics is None:
        logger.warning(f"Could not load graphic asset: {graphics_path}")
        
    frame_graphics1 = cv.imread(graphics1_path, cv.IMREAD_UNCHANGED)
    if frame_graphics1 is None:
        logger.warning(f"Could not load graphic asset: {graphics1_path}")
    
    return frame_graphics, frame_graphics1