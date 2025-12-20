from pydantic import BaseModel, Field, field_validator
from typing import List, Dict
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelConfig(BaseModel):
    name: str
    fallback: str

class VideoConfig(BaseModel):
    source: str
    mask: str
    output: str

class DetectionConfig(BaseModel):
    target_classes: List[str]
    all_classes: Dict[int, str]

    @field_validator('all_classes', mode='before')
    def parse_all_classes(cls, v):
        # Allow input as list, convert to dict {index: name}
        if isinstance(v, list):
            return {i: name for i, name in enumerate(v)}
        return v

class TrackerConfig(BaseModel):
    max_age: int = 20
    min_hits: int = 3
    iou_threshold: float = 0.3

class CountingLinesConfig(BaseModel):
    up: List[int] = Field(..., min_length=4, max_length=4)
    down: List[int] = Field(..., min_length=4, max_length=4)

    @field_validator('up', 'down')
    def validate_coordinates(cls, v):
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Coordinates must be numbers")
        return v

class AssetsConfig(BaseModel):
    graphics_total: str
    graphics_vehicle: str

class AppConfig(BaseModel):
    model: ModelConfig
    video: VideoConfig
    detection: DetectionConfig
    tracker: TrackerConfig
    counting_lines: CountingLinesConfig
    assets: AssetsConfig

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> 'AppConfig':
        """Load and validate configuration from a YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(path, "r") as f:
                raw_config = yaml.safe_load(f)
            return cls(**raw_config)
        except Exception as e:
            logger.error(f"Failed to parse configuration: {e}")
            raise
