"""Blender scene interface for the SDG."""

import bpy
import os
import glob
from sdg_engine.core.model import Scene, Snapshot
from sdg_engine.config import SceneConfig
from sdg_engine.core.interfaces.blender.object import BlenderElement, BlenderLight

from warnings import warn
from typing import List, Optional

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
        
        # REMOVIDO DAQUI: self._setup_world_nodes_for_background()
        # A configuração inicial dos nós será feita em from_scene_config ou diretamente em set_background_image se necessário.

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
        self.axis[0].set_rotation(rotation=(snapshot.yaw, snapshot.roll, 0))
        self.cameras[0].set_location(location=(0, 0, snapshot.camera_height))
        self.lights[0].set_energy(energy=snapshot.light_energy)
        
    def _setup_world_nodes_for_background(self):
        """
        Garante que o mundo tenha um setup de nós para background de imagem:
        Environment Texture -> Background -> World Output.
        Ele tenta encontrar os nós existentes; se não existirem, os cria.
        Retorna o Environment Texture Node e o Background Node.
        """
        world = self.blender_scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            self.blender_scene.world = world

        world.use_nodes = True
        world_nodes = world.node_tree.nodes
        world_links = world.node_tree.links

        # Tenta encontrar os nós existentes
        environment_texture_node = None
        background_node = None
        output_node = None

        for node in world_nodes:
            if node.type == 'SHADER_NODE_TEX_ENVIRONMENT':
                environment_texture_node = node
            elif node.type == 'SHADER_NODE_BACKGROUND':
                background_node = node
            elif node.type == 'SHADER_NODE_OUTPUT_WORLD':
                output_node = node
        
        # Se algum nó essencial estiver faltando, remove TUDO e recria
        if not (environment_texture_node and background_node and output_node):
            warn("World nodes for background not found or incomplete. Recreating them.")
            for node in list(world_nodes): # Remove todos se o setup não estiver completo
                world_nodes.remove(node)

            background_node = world_nodes.new(type='ShaderNodeBackground')
            environment_texture_node = world_nodes.new(type='ShaderNodeTexEnvironment')
            output_node = world_nodes.new(type='ShaderNodeOutputWorld')

            # Posiciona os nós para melhor visualização (opcional)
            environment_texture_node.location = -300, 0
            background_node.location = 0, 0
            output_node.location = 300, 0

            # Conecta os nós
            world_links.new(environment_texture_node.outputs['Color'], background_node.inputs['Color'])
            world_links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
        else:
            # Garante as conexões se os nós já existirem
            if not world_links.find(environment_texture_node.outputs['Color'], background_node.inputs['Color']):
                world_links.new(environment_texture_node.outputs['Color'], background_node.inputs['Color'])
            if not world_links.find(background_node.outputs['Background'], output_node.inputs['Surface']):
                world_links.new(background_node.outputs['Background'], output_node.inputs['Surface'])


        self.blender_scene.render.film_transparent = False

        return environment_texture_node, background_node


    def set_background_image(self, image_filepath: str):
        """
        Carrega uma imagem e a define no nó Environment Texture do mundo.
        Ajusta a resolução de renderização da cena para coincidir com a imagem.
        Garante que os nós do mundo estão configurados corretamente antes de tentar atribuir a imagem.
        """
        environment_texture_node, _ = self._setup_world_nodes_for_background() # Chama aqui para garantir o setup

        # Carrega a imagem atual como fundo
        try:
            img = bpy.data.images.load(image_filepath)
            environment_texture_node.image = img
            print(f"Background image loaded: {image_filepath}")
            
            # Ajusta a resolução de renderização para coincidir com a imagem de fundo
            self.blender_scene.render.resolution_x = img.size[0]
            self.blender_scene.render.resolution_y = img.size[1]
            self.blender_scene.render.resolution_percentage = 100 # Garante 100% da resolução definida

        except Exception as e:
            warn(f"Could not load background image {image_filepath}: {e}")
            # Se a imagem não carregar, o nó permanecerá sem imagem, resultando em preto ou na cor padrão do Background node.
            # O Blender normalmente mostra rosa para imagem não encontrada.
            # Podemos tentar definir para uma cor sólida em caso de falha de carregamento:
            self.set_solid_background_color() # Define para cor sólida em caso de erro de imagem


    def set_solid_background_color(self, color: tuple[float, float, float] = (0.05, 0.05, 0.05)):
        """
        Configura o background do mundo para uma cor sólida.
        Desativa o uso de nós no mundo e define a cor.
        """
        world = self.blender_scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            self.blender_scene.world = world
        
        world.use_nodes = False # Desativa os nós para usar a cor de fundo padrão
        world.color = color
        self.blender_scene.render.film_transparent = False


    @classmethod
    def from_scene_config(
        cls,
        scene_config: SceneConfig,
    ):
        """Create a BlenderScene object from an existing scene."""
        # Load the scene from the file path
        bpy.ops.wm.open_mainfile(filepath=scene_config.scene_path) # Usar bpy.ops.wm.open_mainfile diretamente aqui

        # Map the scene configuration to the Blender objects
        attr_to_names = {
            "cameras": scene_config.camera_names,
            "axis": scene_config.axis_names,
            "elements": scene_config.element_names,
            "lights": scene_config.light_names,
        }
        # Initialize and populate the attributes
        attributes = {
            "name": scene_config.scene_name,
            "blender_scene": bpy.data.scenes[scene_config.scene_name],
        }
        for attr, names in attr_to_names.items():
            attributes[attr] = cls._get_blender_objects(names=names, object_type=attr)

        instance = cls(**attributes) # Calls __init__

        # REMOVIDO DAQUI: instance._setup_world_nodes_for_background()
        # O setup agora é chamado por set_background_image para maior robustez, ou set_solid_background_color.

        return instance

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
        # Esta função foi integrada diretamente em from_scene_config para melhor controle
        # Mantendo-a como um staticmethod para consistência, mas não será mais chamada separadamente em from_scene_config
        # A chamada bpy.ops.wm.open_mainfile(filepath=scene_path) é feita diretamente em from_scene_config agora.
        bpy.ops.wm.open_mainfile(filepath=scene_path)