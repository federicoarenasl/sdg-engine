from sdg_engine.core.interfaces.blender.object import BlenderElement
from sdg_engine.core.interfaces.blender.scene import BlenderScene
from sdg_engine.core.model import Annotation, Snapshot, Dataset
from typing import List, Tuple, Optional
import numpy as np
import bpy

from PIL import Image, ImageDraw, ImageFont

import mathutils


def create_bounding_box(
    scene: BlenderScene,
    camera: BlenderElement,
    element: BlenderElement,
    relative: bool = True,
    resolution: Optional[Tuple[int, int]] = None,
    check_visibility: bool = False,
) -> Optional[np.ndarray]:
    """Create bounding boxes for the elements in the scene.

    Parameters:
    -----------
    scene: BlenderScene
        The scene to create bounding boxes for.
    camera: BlenderElement
        The camera to create bounding boxes for.
    element: BlenderElement
        The element to create bounding boxes for.
    relative: bool
        Whether to return the bounding box in relative coordinates.
    resolution: Optional[Tuple[int, int]]
        The resolution to normalize the bounding box to.
    check_visibility: bool
        Whether to only include visible vertices (not occluded by other objects).

    Returns:
    --------
    Optional[np.ndarray]
        The bounding boxes for the elements in the scene, or None if not in view.
    """
    if check_visibility:
        return create_visible_bounding_box(
            scene, camera, element, relative, resolution
        )
    
    # Prepare transformation matrix from camera to object
    inverse_matrix = camera.get_matrix(inverse=True, normalized=True)
    mesh = element.get_mesh(preserve_all_data_layers=True)

    # Apply transformation matrix to mesh, where the mesh is
    # first transformed to the object's coordinate space,
    mesh.transform(element.get_matrix())
    # then to the camera's coordinate space
    mesh.transform(inverse_matrix)

    # Prepare camera frame 3D coordinates
    frame = [-v for v in camera.object.data.view_frame(scene=scene.blender_scene)[:3]]

    # Then get the object's mesh vertices 2D coordinates
    # in the camera's coordinate space
    lx, ly = calculate_normalized_coordinates(mesh, frame)

    # Finally, compute the bounding box from the 2D coordinates
    return compute_bounding_box(lx, ly, relative=relative, resolution=resolution)


def create_visible_bounding_box(
    scene: BlenderScene,
    camera: BlenderElement,
    element: BlenderElement,
    relative: bool = True,
    resolution: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    """Create bounding boxes for only the visible parts of elements in the scene.
    
    This function uses ray casting to determine which vertices are actually visible
    from the camera's perspective, accounting for occlusion by other objects.

    Parameters:
    -----------
    scene: BlenderScene
        The scene to create bounding boxes for.
    camera: BlenderElement
        The camera to create bounding boxes for.
    element: BlenderElement
        The element to create bounding boxes for.
    relative: bool
        Whether to return the bounding box in relative coordinates.
    resolution: Optional[Tuple[int, int]]
        The resolution to normalize the bounding box to.

    Returns:
    --------
    Optional[np.ndarray]
        The bounding boxes for only visible parts of the elements, or None if not visible.
    """
    # Get the depsgraph for ray casting
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    # Get mesh and prepare transformations
    mesh = element.get_mesh(preserve_all_data_layers=True)
    element_matrix = element.get_matrix()
    camera_location = camera.object.matrix_world.translation
    
    # Find visible vertices using ray casting
    visible_vertices = []
    
    for vertex in mesh.vertices:
        # Transform vertex to world coordinates
        world_vertex = element_matrix @ vertex.co
        
        # Cast ray from camera to vertex
        direction = (world_vertex - camera_location).normalized()
        ray_distance = (world_vertex - camera_location).length
        
        # Perform ray cast
        result, location, normal, index, hit_object, matrix = scene.blender_scene.ray_cast(
            depsgraph, camera_location, direction, distance=ray_distance + 0.001
        )
        
        # Check if the ray hit our target object first (not occluded)
        if result and hit_object == element.object:
            # Check if the hit location is close to our vertex
            distance_to_vertex = (location - world_vertex).length
            if distance_to_vertex < 0.01:  # Small tolerance for floating point errors
                visible_vertices.append(vertex)
        elif not result:
            # No hit means the vertex is visible (nothing blocking it)
            visible_vertices.append(vertex)
    
    if not visible_vertices:
        return None
    
    # Now process visible vertices similar to the original function
    inverse_matrix = camera.get_matrix(inverse=True, normalized=True)
    
    # Create a temporary mesh with only visible vertices
    visible_mesh = bpy.data.meshes.new("temp_visible_mesh")
    visible_mesh.from_pydata(
        [vertex.co for vertex in visible_vertices], 
        [], 
        []
    )
    
    # Apply transformations
    visible_mesh.transform(element_matrix)
    visible_mesh.transform(inverse_matrix)
    
    # Prepare camera frame 3D coordinates
    frame = [-v for v in camera.object.data.view_frame(scene=scene.blender_scene)[:3]]
    
    # Calculate normalized coordinates for visible vertices
    lx, ly = calculate_normalized_coordinates(visible_mesh, frame)
    
    # Clean up temporary mesh
    bpy.data.meshes.remove(visible_mesh)
    
    # Finally, compute the bounding box from the 2D coordinates
    return compute_bounding_box(lx, ly, relative=relative, resolution=resolution)


def calculate_normalized_coordinates(
    mesh: bpy.types.Mesh, frame: List[mathutils.Vector]
) -> Tuple[List[float], List[float]]:
    """Calculate the normalized coordinates of the mesh vertices.

    Parameters:
    -----------
    mesh: bpy.types.Mesh
        The mesh to calculate coordinates for.
    frame: List[mathutils.Vector]
        The camera frame to use for calculations.

    Returns:
    --------
    Tuple[List[float], List[float]]
        The normalized x and y coordinates of the mesh vertices.
    """
    lx = []
    ly = []

    for v in mesh.vertices:
        co_local = v.co
        z = -co_local.z

        if z <= 0.0:
            continue
        else:
            frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    return lx, ly


def compute_bounding_box(
    lx: List[float],
    ly: List[float],
    relative: bool = True,
    resolution: Tuple[int, int] = (256, 256),
) -> Optional[np.ndarray]:
    """Compute the bounding box from the normalized coordinates.

    The returned bounding box is in the format (top_x, top_y, width, height)
    measured from the top left corner of the image.

    Parameters:
    -----------
    lx: List[float]
        The list of normalized x coordinates for a given mesh's vertices.
    ly: List[float]
        The list of normalized y coordinates for a given mesh's vertices.
    relative: bool
        Whether to return the bounding box in relative coordinates.
    resolution: Tuple[int, int]
        The resolution to normalize the bounding box to.

    Returns:
    --------
    Optional[np.ndarray]
        The computed bounding box, or None if not in view.
        The format returned is (top_x, top_y, width, height)
    """
    if not lx or not ly:
        return None

    # Flip y-coordinates if they are inverted (assuming normalized [0, 1])
    ly = [1 - y for y in ly]

    # Calculate the bounding box limits
    top_x, bottom_x = np.clip([min(lx), max(lx)], 0.0, 1.0)
    top_y, bottom_y = np.clip([min(ly), max(ly)], 0.0, 1.0)

    # Check if the bounding box is valid
    if top_x == bottom_x or top_y == bottom_y:
        return None

    # Convert to relative coordinates if required
    if relative:
        top_x, top_y, bottom_x, bottom_y = make_bounding_box_relative(
            top_x, top_y, bottom_x, bottom_y, resolution
        )

    # Calculate width and height
    width = bottom_x - top_x
    height = bottom_y - top_y

    # Return the bounding box as a numpy array
    return np.array([top_x or 1e-6, top_y or 1e-6, width, height])


def make_bounding_box_relative(
    top_x: float,
    top_y: float,
    bottom_x: float,
    bottom_y: float,
    resolution: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """Make the bounding box relative to the resolution.

    Parameters:
    -----------
    top_x: float
        The top x coordinate of the bounding box.
    top_y: float
        The top y coordinate of the bounding box.
    bottom_x: float
        The bottom x coordinate of the bounding box.
    bottom_y: float
        The bottom y coordinate of the bounding box.
    resolution: Tuple[int, int]
        The resolution to normalize the bounding box to.

    Returns:
    --------
    Tuple[float, float, float, float]
        The bounding box in relative coordinates.

    Raises:
    -------
    ValueError
        If the resolution is not provided.
    """
    if resolution is None:
        raise ValueError("Resolution must be provided if relative is True")

    return (
        top_x * resolution[0],
        top_y * resolution[1],
        bottom_x * resolution[0],
        bottom_y * resolution[1],
    )


def draw_bounding_box_with_category(
    target_path: str, annotation: Annotation, snapshot: Snapshot
) -> None:
    """
    Draws bounding boxes and category labels on the image specified by the image_path.

    Parameters:
    -----------
    target_path: str
        The path to the image file.
    annotation: Annotation
        The annotation to draw on the image.

    """
    # Open the image
    image = Image.open(f"{target_path}/{annotation.file_name}")
    draw = ImageDraw.Draw(image)

    # Extract bounding box and category data
    bboxes = annotation.objects.bbox
    categories = annotation.objects.categories

    # Define a font for the category label
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Draw debugging rectangle, and Annotation JSON inside it
    draw.text((0, 0), "(0, 0)", fill="white", font=font)
    draw.text(
        (0, 30),
        f"Snapshot: {snapshot.model_dump_json(indent=4)}",
        fill="white",
        font=font,
    )

    # Draw each bounding box and category label
    for bbox, category in zip(bboxes, categories):
        top_x, top_y, width, height = bbox

        # Calculate the bottom coordinates
        bottom_x = top_x + width
        bottom_y = top_y + height

        # Draw the rectangle
        draw.rectangle([top_x, top_y, bottom_x, bottom_y], outline="red", width=2)

        # Draw the category label
        label = f"{category}, ({top_x:.2f}, {top_y:.2f})"
        text_size = draw.textbbox((0, 0), label, font=font)[2:]
        text_background = [top_x, top_y - text_size[1], top_x + text_size[0], top_y]
        draw.rectangle(text_background, fill="red")
        draw.text((top_x, top_y - text_size[1]), label, fill="white", font=font)

    # Save the image to the same path but with the suffix "_annotated"
    image.save(
        f"{target_path}/{annotation.file_name.replace('.png', '')}_annotated.png"
    )


def render_annotation_animation(target_path: str, dataset: Dataset) -> None:
    """
    Creates an animation GIF from all annotated images in the dataset.

    Parameters:
    -----------
    target_path: str
        The path where the annotated images are stored.
    dataset: Dataset
        The dataset containing the annotations.
    """
    # Get all annotated image paths
    image_paths = [
        f"{target_path}/{annotation.file_name.replace('.png', '')}_annotated.png"
        for annotation in dataset.annotations
    ]
    
    # Open all images
    images = [Image.open(path) for path in image_paths]
    
    # Save as GIF
    output_path = f"{target_path}/annotation_animation.gif"
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=500,  # 500ms per frame
        loop=0  # Loop indefinitely
    )
    
    # Close all images
    for img in images:
        img.close()
