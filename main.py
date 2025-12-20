import cv2 as cv
import cvzone
import argparse
import logging
import sys
import os

from src import (
    AppConfig,
    initialize_model,
    load_video_and_mask,
    initialize_video_writer,
    load_graphic_assets,
    detect_vehicles,
    VehicleCounter,
    Sort
)

try:
    import google.colab
    IN_COLAB = True
    from IPython.display import display, clear_output
    from PIL import Image
except ImportError:
    IN_COLAB = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Tracking and Counting System")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--mask", type=str, help="Path to mask image")
    parser.add_argument("--output", type=str, help="Path to output video file")
    parser.add_argument("--model", type=str, help="YOLO model path (e.g., yolov8n.pt)")
    parser.add_argument("--no-display", action="store_true", help="Disable real-time video display")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose (DEBUG) logging")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after processing N frames (0 for infinite)")
    
    # In Colab/Notebooks, sys.argv might contain specific arguments we don't handle.
    if IN_COLAB:
        args, _ = parser.parse_known_args()
        return args
    else:
        return parser.parse_args()

def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        config = AppConfig.load(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        exit(1)

    if IN_COLAB:
        logger.info("Running in Colab environment. Using default Colab paths.")
        video_source = "/content/assets/traffic_cam.mp4"
        mask_path = "/content/assets/mask.png"
        output_path = "/content/result.mp4"
        graphics_total = "/content/assets/graphics.png"
        graphics_vehicle = "/content/assets/graphics1.png"
    else:
        video_source = args.video if args.video else config.video.source
        mask_path = args.mask if args.mask else config.video.mask
        output_path = args.output if args.output else config.video.output
        graphics_total = config.assets.graphics_total
        graphics_vehicle = config.assets.graphics_vehicle

    model_name = args.model if args.model else config.model.name

    model = initialize_model(model_name, config.model.fallback)

    vid, mask = load_video_and_mask(video_source, mask_path)
    if vid is None:
        logger.error("Failed to initialize video. Exiting.")
        exit(1)

    video_writer = initialize_video_writer(vid, output_path)
    if video_writer is None:
        logger.error("Failed to initialize video writer. Exiting.")
        vid.release()
        exit(1)

    class_names = config.detection.all_classes
    target_classes = config.detection.target_classes

    tracker = Sort(
        max_age=config.tracker.max_age, 
        min_hits=config.tracker.min_hits, 
        iou_threshold=config.tracker.iou_threshold
    )

    line_up = config.counting_lines.up
    line_down = config.counting_lines.down
    
    counter = VehicleCounter(line_up, line_down)

    frame_graphics, frame_graphics1 = load_graphic_assets(graphics_total, graphics_vehicle)

    logger.info(f"Starting video processing loop... (Output: {output_path})")
    
    frame_num = 0
    display_interval = 20

    try:
        while True:
            ret, frame = vid.read()
            if not ret:
                logger.info("End of video or error reading frame. Finishing...")
                break
            
            frame_num += 1

            # Check if max frames reached
            if args.max_frames > 0 and frame_num > args.max_frames:
                logger.info(f"Reached maximum frames ({args.max_frames}). Finishing...")
                break

            # Apply mask if available
            if mask is not None:
                frame_region = cv.bitwise_and(frame, mask)
            else:
                frame_region = frame

            detections = detect_vehicles(model, frame_region, class_names, target_classes)

            if frame_graphics is not None:
                frame = cvzone.overlayPNG(frame, frame_graphics, (0,0))

            if frame_graphics1 is not None:
                frame = cvzone.overlayPNG(frame, frame_graphics1, (420,0))

            tracker_updates = tracker.update(detections)
            cv.line(frame, (line_up[0], line_up[1]), (line_up[2], line_up[3]), (0, 0, 255), thickness = 3)
            cv.line(frame, (line_down[0] ,line_down[1]), (line_down[2], line_down[3]), (0, 0, 255), thickness = 3)

            for update in tracker_updates:
                x1, y1, x2, y2, id = update
                x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
                w, h = (x2-x1), (y2-y1)

                cx, cy = (x1+w//2), (y1+h//2)
                cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)

                counter.update(cx, cy, id)

                cvzone.cornerRect(frame, (x1, y1, w, h), l=5, colorR=(255, 0, 255), rt=1)
                cvzone.putTextRect(frame, f'{id}', (x1, y1), scale=1, thickness=2)

            count_up_list, count_down_list, total_count_list = counter.get_counts()

            cv.putText(frame, str(len(total_count_list)), (255, 100), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), thickness=7)
            cv.putText(frame, str(len(count_up_list)), (600, 85), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), thickness=7)
            cv.putText(frame, str(len(count_down_list)), (850, 85), cv.FONT_HERSHEY_PLAIN, 5, (200, 50, 200), thickness=7)

            video_writer.write(frame)

            if IN_COLAB:
                if frame_num % display_interval == 0:
                    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    clear_output(wait=True)
                    display(Image.fromarray(rgb))
            elif not args.no_display:
                cv.imshow("vid", frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested exit.")
                    break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected. Stopping...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        vid.release()
        try:
            cv.destroyAllWindows()
        except Exception:
            pass
        video_writer.release()
        logger.info("Cleanup complete.")

if __name__ == "__main__":
    main()
