"""Blender scene interface for the SDG."""

import bpy
from sdg_engine.core.model import Scene, Snapshot
from sdg_engine.config import SceneConfig
from sdg_engine.core.interfaces.blender.object import BlenderElement, BlenderLight

from warnings import warn
from typing import List


class BlenderScene(Scene):
    """Interface for Blender scenes."""

    blender_scene: bpy.types.Scene

    def __init__(
        self,
        blender_scene: bpy.types.Scene,
        **kwargs,
    ):
        """Initialize the interface with a Blender scene."""
        super().__init__(**kwargs, blender_scene=blender_scene)

    def prepare_from_snapshot(self, snapshot: Snapshot):
        """Prepare the scene for a snapshot."""
        for attr, message in [
            ("axis", "axes"),
            ("cameras", "cameras"),
            ("lights", "lights"),
        ]:
            if len(getattr(self, attr)) > 1:
                warn(f"Multiple {message} found in the scene. Using the first one.")

        # Prepare axis, camera and light
        self.axis[0].set_location(location=(0, 0, 0))
        self.axis[0].set_rotation(rotation=(snapshot.roll, 0, snapshot.yaw))
        self.cameras[0].set_location(location=(0, 0, snapshot.camera_height))
        self.lights[0].set_energy(energy=snapshot.light_energy)

    @classmethod
    def from_scene_config(
        cls,
        scene_config: SceneConfig,
    ):
        """Create a BlenderScene object from an existing scene."""
        # Load the scene from the file path
        cls._load_from_scene_path(scene_config.scene_path)

        # Map the scene configuration to the Blender objects
        attr_to_names = {
            "cameras": scene_config.camera_names,
            "axis": scene_config.axis_names,
            "elements": list(scene_config.element_mapping.keys()),
            "lights": scene_config.light_names,
        }
        # Initialize and populate the attributes
        attributes = {
            "name": scene_config.scene_name,
            "blender_scene": bpy.data.scenes[scene_config.scene_name],
        }
        for attr, names in attr_to_names.items():
            attributes[attr] = cls._get_blender_objects(names=names, object_type=attr)

        return cls(**attributes)

    @staticmethod
    def _get_blender_objects(
        object_type: str, names: List[str]
    ) -> list[BlenderElement]:
        """Get Blender objects from a list of names and their type."""

        if object_type == "lights":
            return [
                BlenderLight.from_bpy_object(bpy.data.objects[name]) for name in names
            ]
        return [
            BlenderElement.from_bpy_object(bpy.data.objects[name]) for name in names
        ]

    @staticmethod
    def _load_from_scene_path(scene_path: str) -> bpy.types.Scene:
        bpy.ops.wm.open_mainfile(filepath=scene_path)
