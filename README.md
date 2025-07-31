# sdg-engine: a simple data generation engine for computer vision

A simple interface for synthetic data generation for computer vision using rendering engines.

By following some conventions (more below), you can basically set up any scene in Blender and `sdg-engine` will output an annotated dataset importable into 🤗  HuggingFace `datasets` to be used in your downstream Computer Vision tasks.

Enjoy simple synthetic data generation!

> Notes/Disclaimers:
> - Currently only interfaces with Blender, but should be extendable to others.
> - This is a work in progress and the interface might change.

## Installation
Clone this repository:
```bash
git clone https://github.com/federicoarenasl/sdg-engine.git
```
Install dependencies using `poetry` (if you don't have poetry you can easily install following the steps [here](https://python-poetry.org/docs/#installing-with-pipx)):
```bash
poetry install
```
This will create a virtual environment locally at `.venv`, simply activate this before running the project.
```bash
source .venv/bin/activate
```

> Note: Currently Blender's `bpy` package is included as part of the dependencies. Depending on your internet speed, this will take a short while.

## Running `sdg_engine` using Blender
In order for `sdg_engine` to remain minimal while illustrative for you to generate your own synthetic datasets, I've included a functional example.

You can also use your own scenes by following the instructions below.
### From the example included
This is composed of two files:
- [`gelatina.blend`](gelatina.blend): This is the actual Blender `.blend` scene.
- [`config.yaml`](config.yaml): This is the configuration for that indicates `sdg_engine` which `.blend` scene to use, where to save the data, and how to perform a sweep over the scene in order to generate meaningful data.


### From your own example
You'll have to follow some conventions I've setup to make the actual setup super easy.

Open Blender (if you haven't installed it yet, you can do so [here](https://www.blender.org/download/)) and:
1. Edit the scene [`gelatina.blend`] with the objects that you want to generate and other customizations. 
2. In your new scene, make sure to name your objects with easily identifiable names. Don`t forget to put the exact names of the objects in the yaml.
3. Spend as much time as you would like styling your scene to your liking.
4. Keep the camera and axis as they are in the example scene, **do not delete nor move these**.

Update the configuration YAML  will use to generate data:
1. Using the `config.yaml`, replace `scene_path` with the path to your new `.blend` file.
2. Update the `scene_name` to the name of the scene in your `.blend` file.
3. Update the `element_names` to the names of the objects in your scene.
4. Update the `target_path` to the path where you want to save the data.
5. Finally, put in the 'background' folder path which contains the backgrounds that you want to generate the data with

### Generating the dataset
Once you have your scene set up, you can generate data by running:
```bash
poetry run python -m sdg_engine.main --config config.yaml
```

This will produce a dataset with the following structure in your `target_path`:
```bash
.
└── train # or validation or test, depending on the split in the config
    ├── images
      ├── 1ed8d595-fb93-4d7b-87bc-d060a66a0b66_annotated.png
      ├── 1ed8d595-fb93-4d7b-87bc-d060a66a0b66.png
      ├── 4a283977-67aa-4485-aefe-4fcbf3a74731_annotated.png
      ...
      ├── f3b8cc29-de80-4b52-961c-a00ec237767c_annotated.png
      ├── f3b8cc29-de80-4b52-961c-a00ec237767c.png
      ├── annotation_animation.gif
      
    └── labels
        ├── 1ed8d595-fb93-4d7b-87bc-d060a66a0b66_annotated.txt
      ├── 1ed8d595-fb93-4d7b-87bc-d060a66a0b66.txt
      ├── 4a283977-67aa-4485-aefe-4fcbf3a74731_annotated.txt
      ...
      ├── f3b8cc29-de80-4b52-961c-a00ec237767c_annotated.txt
      ├── f3b8cc29-de80-4b52-961c-a00ec237767c.txt
    └── metadata.jsonl
```

By default, the `config.yaml` has debug mode enabled, this will:
- save an additional image with the rendered bounding boxes on top of the original image, hence the `_annotated` suffix.
- save an animation of the annotated images at `target_path/<split>/annotation_animation.gif`.

The animation GIF is super helpful to visualize the data you've just generated, see it
below for the `example.config.yaml` configuration.

<p align="center">
  <img src="https://github.com/federicoarenasl/sdg-engine/blob/main/examples/annotation_animation.gif" width="350" />
</p>
<p align="center">
  <i>Animation of the annotated images generated with the example configuration.</i>
</p>


## Training Yolo with the dataset generated

To train the yolo with the dataset generated you need to:
1. Put a test folder with images the same size as the images in the train folder
2. Edit the config_yolo.yaml and train.py file with the yolo configs that you want
3. Run the training with:
            poetry run train.py

## Pushing your dataset to the 🤗 Hub

Once you have your dataset saved, you can push it to the 🤗 Hub by running:
```python
from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="path/to/your/dataset")
dataset.push_to_hub("your-username/your-dataset-name")
```

I've included an example notebook in the `examples` folder to push the dataset to the 🤗 Hub, check it out at [`examples/push-dataset-to-hub.ipynb`](examples/push-dataset-to-hub.ipynb).


## Tutorials
There will be a tutorial on how to use the generated dataset in a downstream Computer Vision task using 🤗 Transformers and 🤗 Datasets in my website [federicoarenas.ai](https://federicoarenas.ai).

Stay tuned for the tutorial at [federicoarenas.ai/projects/sdg-engine-applied](https://federicoarenas.ai/projects/sdg-engine-applied)


## Contributing

I'm open to contributions! Please feel free to open an issue or a PR.



