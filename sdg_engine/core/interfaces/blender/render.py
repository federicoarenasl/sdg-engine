import bpy
import math
import os
import glob
import random
import warnings
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional
import uuid

from sdg_engine.core.interfaces.blender.scene import BlenderScene
from sdg_engine.core.interfaces.blender.sweep import BlenderSweep
from sdg_engine.core.interfaces.blender.object import BlenderElement
from sdg_engine.config import RenderingConfig
from sdg_engine.core.interfaces.blender import utils # Esta é a importação crucial

from sdg_engine.core.model import Dataset, Annotation, SnapshotAnnotation

METADATA_FILENAME = "metadata.jsonl"

class BlenderRenderer:
    def __init__(
        self,
        scene: BlenderScene,
        target_path: str,
        resolution: Tuple[int, int],
        samples: int,
    ):
        self.scene = scene
        self.target_path = target_path
        self.resolution = resolution
        self.scene.blender_scene.render.resolution_x = resolution[0]
        self.scene.blender_scene.render.resolution_y = resolution[1]
        self.scene.blender_scene.cycles.samples = samples

    @classmethod
    def from_scene(
        cls,
        scene: BlenderScene,
        target_path: str,
        resolution: Tuple[int, int],
        samples: int,
    ):
        return cls(scene, target_path, resolution, samples)

    def render_snapshot(
        self,
        snapshot_id: Optional[uuid.UUID] = None,
        custom_filepath: Optional[str] = None
    ) -> str:
        if custom_filepath:
            self.scene.blender_scene.render.filepath = custom_filepath
        elif snapshot_id:
            self.scene.blender_scene.render.filepath = f"{self.target_path}/{snapshot_id}.png"
        else:
            raise ValueError("Either snapshot_id or custom_filepath must be provided for rendering.")
            
        bpy.ops.render.render(write_still=True)
        return self.scene.blender_scene.render.filepath

    def annotate_snapshot(
        self,
        cameras: List[BlenderElement],
        elements: List[BlenderElement],
        snapshot_id: uuid.UUID,
        relative: bool = True,
        file_name_override: Optional[str] = None,
        yolo_output_path: Optional[str] = None
    ) -> Annotation:
        if len(cameras) > 1:
            warnings.warn(
                "Multiple cameras are not supported yet. Only the first camera will be used."
            )
        camera = cameras[0]

        annotation = Annotation(
            file_name=file_name_override if file_name_override else f"{snapshot_id}.png",
            objects=SnapshotAnnotation(bbox=[], categories=[]),
        )

        current_render_resolution = (
            self.scene.blender_scene.render.resolution_x,
            self.scene.blender_scene.render.resolution_y
        )
        img_width, img_height = current_render_resolution

        yolo_lines = []

        for i, element in enumerate(elements):
            # A correção para coordenadas negativas deve ocorrer dentro de utils.create_bounding_box
            bounding_box: np.ndarray = utils.create_bounding_box(
                scene=self.scene,
                camera=camera,
                element=element,
                relative=relative, # Passa 'relative' para a função de bounding box
                resolution=current_render_resolution,
            )
            

            if bounding_box is None:
                continue

            top_x, top_y, width, height = bounding_box.tolist()

            annotation.objects.bbox.append(bounding_box.tolist())
            annotation.objects.categories.append(i)

            center_x = top_x + (width / 2)
            center_y = top_y + (height / 2)

            normalized_center_x = center_x / img_width
            normalized_center_y = center_y / img_height
            normalized_width = width / img_width
            normalized_height = height / img_height

            yolo_line = f"{i} {normalized_center_x:.6f} {normalized_center_y:.6f} {normalized_width:.6f} {normalized_height:.6f}"
            yolo_lines.append(yolo_line)

        if yolo_output_path:
            os.makedirs(os.path.dirname(yolo_output_path), exist_ok=True)
            with open(yolo_output_path, "w") as f:
                for line in yolo_lines:
                    f.write(line + "\n")

        return annotation

def generate_dataset_from_config(config: RenderingConfig) -> Dataset:
    scene: BlenderScene = BlenderScene.from_scene_config(config.scene_config)
    sweep: BlenderSweep = BlenderSweep.from_sweep_config(config.sweep_config)

    split_images_path = os.path.join(config.target_path, config.split, "images")
    split_labels_path = os.path.join(config.target_path, config.split, "labels")

    os.makedirs(split_images_path, exist_ok=True)
    os.makedirs(split_labels_path, exist_ok=True)

    renderer: BlenderRenderer = BlenderRenderer.from_scene(
        scene,
        split_images_path,
        config.resolution,
        config.samples,
    )

    dataset: Dataset = Dataset(path=config.target_path, annotations=[])

    background_images_folder_path = getattr(config.scene_config, 'background_images_folder_path', None)
    all_background_images = []

    if background_images_folder_path and os.path.isdir(background_images_folder_path):
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp', '*.hdr')
        for ext in image_extensions:
            all_background_images.extend(glob.glob(os.path.join(background_images_folder_path, ext)))
        
        if not all_background_images:
            warnings.warn(f"No background images found in: '{background_images_folder_path}'. Renders will have a default solid color background.")
    elif background_images_folder_path:
        warnings.warn(f"The provided path for backgrounds is not a valid directory: '{background_images_folder_path}'. Renders will have a default solid color background.")
    else:
        warnings.warn("No background images folder path provided in config.scene_config.background_images_folder_path. Renders will have a default solid color background.")
    
    for snapshot_idx, snapshot in enumerate(tqdm(sweep.snapshots, desc="Rendering snapshots")):
        scene.prepare_from_snapshot(snapshot=snapshot)

        backgrounds_to_render = all_background_images if all_background_images else [None]

        for bg_idx, background_filepath in enumerate(backgrounds_to_render):
            if background_filepath:
                print(f"Applying background: {os.path.basename(background_filepath)}")
                scene.set_background_image(background_filepath)
            else:
                scene.set_solid_background_color()
                scene.blender_scene.render.resolution_x = renderer.resolution[0]
                scene.blender_scene.render.resolution_y = renderer.resolution[1]
                scene.blender_scene.render.resolution_percentage = 100

            render_file_base_name = f"{snapshot.id.hex}_{bg_idx:04d}" 
            full_output_filepath_png = os.path.join(renderer.target_path, f"{render_file_base_name}.png")
            full_output_filepath_txt = os.path.join(split_labels_path, f"{render_file_base_name}.txt")
            
            renderer.render_snapshot(custom_filepath=full_output_filepath_png)
            
            annotation: Annotation = renderer.annotate_snapshot(
                cameras=scene.cameras,
                elements=scene.elements,
                snapshot_id=snapshot.id,
                file_name_override=f"{render_file_base_name}.png",
                yolo_output_path=full_output_filepath_txt
            )

            if config.debug:
                utils.draw_bounding_box_with_category(
                    target_path=split_images_path,
                    annotation=annotation,
                    snapshot=snapshot,
                )

            dataset.annotations.append(annotation)

    if config.debug:
        utils.render_annotation_animation(split_images_path, dataset)

    with open(f"{config.target_path}/{METADATA_FILENAME}", "w") as f:
        for annotation in dataset.annotations:
            f.write(annotation.model_dump_json() + "\n")

    return dataset