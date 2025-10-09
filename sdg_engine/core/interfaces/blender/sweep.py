from sdg_engine.core.model import Sweep, Snapshot
from sdg_engine.config import SweepConfig
from itertools import product
from typing import List


class BlenderSweep(Sweep):
    """Interface for Blender sweeps."""

    def __init__(self, **kwargs):
        """Initialize the interface with a Blender sweep."""
        super().__init__(**kwargs)

    @classmethod
    def from_sweep_config(cls, sweep_config: SweepConfig):
        """Create a BlenderSweep object from an existing sweep."""
        return cls(
            name=sweep_config.name,
            snapshots=cls.snapshots_from_config(sweep_config),
        )

    @staticmethod
    def snapshots_from_config(sweep_config: SweepConfig) -> List[Snapshot]:
        """Generate snapshots from a sweep configuration.
        
        The 'step' parameter represents the number of intervals to divide each parameter range into.
        This ensures proportional sampling across all parameters regardless of their scale:
        
        - For angles (yaw: -180 to 180°, roll: -85 to 85°): step=8 gives reasonable angular increments
        - For camera height (e.g., 1.0 to 2.0): step=8 gives fine height increments  
        - For light energy (e.g., 1000 to 5000): step=8 gives appropriate energy increments
        
        Each parameter range is divided into 'step' equal intervals, creating step+1 values
        (including both endpoints).
        
        Example with step=4:
        - yaw_limits [-180, 180] → values: [-180, -90, 0, 90, 180] (5 values)
        - camera_height_limits [1.0, 2.0] → values: [1.0, 1.25, 1.5, 1.75, 2.0] (5 values)
        """
        import numpy as np
        
        # Generate evenly spaced values for each parameter using the step as number of intervals
        yaw_values = np.unique(np.linspace(
            sweep_config.yaw_limits[0], 
            sweep_config.yaw_limits[1], 
            sweep_config.step + 1
        ))
        print(f"[DEBUG] Yaw values: {yaw_values}")
        
        roll_values = np.unique(np.linspace(
            sweep_config.roll_limits[0], 
            sweep_config.roll_limits[1], 
            sweep_config.step + 1
        ))
        print(f"[DEBUG] Roll values: {roll_values}")
        camera_height_values = np.unique(np.linspace(
            sweep_config.camera_height_limits[0], 
            sweep_config.camera_height_limits[1], 
            sweep_config.step + 1
        ))
        print(f"[DEBUG] Camera height values: {camera_height_values}")
        light_energy_values = np.unique(np.linspace(
            sweep_config.light_energy_limits[0], 
            sweep_config.light_energy_limits[1], 
            sweep_config.step + 1
        ))
        print(f"[DEBUG] Light energy values: {light_energy_values}")
        return [
            Snapshot(
                yaw=float(yaw),
                roll=float(roll),
                camera_height=float(camera_height),
                light_energy=float(light_energy),
            )
            for yaw, roll, camera_height, light_energy in product(
                yaw_values, roll_values, camera_height_values, light_energy_values
            )
        ]
