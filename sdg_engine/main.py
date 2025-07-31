from sdg_engine.config import RenderingConfig, SupportedEngines, config_from_yaml

import os


def generate_dataset(config: RenderingConfig):
    """Load a scene from a YAML configuration and trigger the rendering sweep."""
    # Check if the engine is supported
    if config.engine != SupportedEngines.BLENDER:
        raise ValueError(f"Unsupported engine: {config.engine}")

    # Import the renderer and generate the dataset
    import sdg_engine.core.interfaces.blender.render as renderer

    return renderer.generate_dataset_from_config(config)



if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Render a sweep of snapshots.")
    parser.add_argument(
        "--config_path", type=str, help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    # Load the configuration and render the sweep
    with open(args.config_path, "r") as config_file:
        config = config_from_yaml(yaml.load(config_file, Loader=yaml.FullLoader))

    # Render the sweep
    generate_dataset(config)
