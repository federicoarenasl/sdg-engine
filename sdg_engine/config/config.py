from typing import List, Tuple, Dict
from pydantic import BaseModel
from enum import Enum


class SupportedEngines(str, Enum):
    """Supported rendering engines."""

    BLENDER = "blender"


class SceneConfig(BaseModel):
    scene_name: str
    scene_path: str
    background_images_folder_path: str
    camera_names: List[str]
    axis_names: List[str]
    element_names: List[str]
    light_names: List[str]


class SweepConfig(BaseModel):
    name: str
    step: int
    yaw_limits: Tuple[float, float]
    roll_limits: Tuple[float, float]
    camera_height_limits: Tuple[float, float]
    light_energy_limits: Tuple[float, float]


class RenderingConfig(BaseModel):
    random_seed: int
    resolution: Tuple[int, int]
    samples: int
    target_path: str
    split: str
    engine: SupportedEngines
    scene_config: SceneConfig
    sweep_config: SweepConfig
    debug: bool = False


def config_from_yaml(yaml_config: Dict) -> RenderingConfig:
    """Create a rendering configuration from a YAML file."""
    return RenderingConfig(**yaml_config)
