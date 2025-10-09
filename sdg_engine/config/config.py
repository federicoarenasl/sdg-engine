from typing import List, Tuple, Dict
from pydantic import BaseModel
from enum import Enum


class SupportedEngines(str, Enum):
    """Supported rendering engines."""

    BLENDER = "blender"


class SceneConfig(BaseModel):
    scene_name: str
    scene_path: str
    camera_names: List[str]
    axis_names: List[str]
    element_mapping: Dict[str, int]
    light_names: List[str]


class SweepConfig(BaseModel):
    """Configuration for parameter sweeps.
    
    The 'step' parameter defines how many intervals to divide each parameter range into,
    ensuring proportional sampling across all parameters regardless of their scale.
    For example, step=8 will create 9 values (8 intervals + endpoints) for each parameter.
    """
    name: str
    step: int  # Number of intervals to divide each parameter range into
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
    check_visibility: bool = False


def config_from_yaml(yaml_config: Dict) -> RenderingConfig:
    """Create a rendering configuration from a YAML file."""
    return RenderingConfig(**yaml_config)
