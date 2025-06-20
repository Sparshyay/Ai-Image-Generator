# AI Image Generator

This project uses Stable Diffusion to generate images from text prompts.

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Make sure you have a CUDA-capable GPU and the necessary NVIDIA drivers installed.

## Usage

You can generate images by running the main script:
```bash
python image_generator.py
```

The script will generate an image based on a default prompt. You can modify the prompt in the `image_generator.py` file or create your own function calls to generate different images.

## Custom Usage

To generate your own images, use the `generate_image()` function:
```python
from image_generator import generate_image

# Generate an image with a custom prompt
generate_image("A beautiful sunset over a mountain landscape", "my_image.png")
```
