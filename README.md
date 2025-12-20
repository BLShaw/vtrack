# VTrack - Industrial-Grade Vehicle Tracking and Counting

VTrack is a high-performance, real-time vehicle detection and counting system built with YOLOv8 and the SORT (Simple Online and Realtime Tracking) algorithm. Remastered for industrial-grade excellence with a modular architecture, robust configuration validation, and unified environment support.

## Key Features

- **Unified Environment Support**: Automatically detects and manages local vs. Google Colab/Notebook environments.
- **Robust Configuration**: Powered by **Pydantic** for strict type-safe validation of `config.yaml`.
- **Advanced Counting Logic**: Uses Vector Cross-Product algorithm for accurate line-crossing detection on any orientation (diagonal/vertical).
- **Industrial Standards**:
  - `src` layout for package integrity.
  - Comprehensive logging instead of print statements.
  - Type hinting throughout the codebase.
  - Full test suite with `pytest`.

## Demo

https://github.com/user-attachments/assets/d506c380-c96d-4e52-9c9a-b918d12d4671

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/BLShaw/vtrack.git
   cd vtrack
   ```

2. **Install in editable mode**:
   ```bash
   pip install -e .
   ```
   *Alternatively, install from requirements.txt:*
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Local Execution
```bash
python main.py --video assets/traffic_cam.mp4 --no-display
```

### Google Colab / Notebook
Simply run `python main.py`. The system will automatically switch to inline image rendering using `IPython.display` and use appropriate Colab paths.

### CLI Arguments
- `--config`: Path to configuration file (default: `config.yaml`).
- `--video`: Override input video path.
- `--output`: Override output video path.
- `--no-display`: Disable real-time video window.
- `--verbose`: Enable DEBUG level logging.
- `--max-frames`: Limit processing to N frames (useful for testing).

## Configuration

All parameters are managed in `config.yaml`:
- **Detection**: Target classes (cars, trucks, etc.) and model selection.
- **Tracker**: SORT parameters (`max_age`, `min_hits`, `iou_threshold`).
- **Counting Lines**: Defined as `[x1, y1, x2, y2]` segments.
- **Assets**: Paths for UI overlays and masks.

## Development

### Testing
Run the comprehensive test suite (Unit, Integration, System, and Acceptance):
```bash
pytest
```

### File Structure
```
VTrack/
├── main.py             # Unified entry point
├── config.yaml          # Type-safe configuration
├── pyproject.toml       # Build and dependency metadata
├── src/                 # Source code package
│   ├── config.py        # Pydantic models
│   ├── counter.py       # Crossing logic
│   ├── detection.py     # YOLO interface
│   ├── tracking.py      # SORT implementation
│   └── utils/           # I/O and Video utilities
├── tests/               # Professional test suite
└── assets/              # Media and UI assets
```

## Acknowledgments
- **Ultralytics YOLOv8** for state-of-the-art detection.
- **SORT Algorithm** by Alex Bewley for robust tracking.
- **Pydantic** for configuration excellence.
