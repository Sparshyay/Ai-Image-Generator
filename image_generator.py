import os
import argparse
import torch
import numpy as np
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline
)
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Union, Tuple
import random
import math

def setup_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    
    # Using a more advanced model for better quality
    model_id = "stabilityai/stable-diffusion-2-1"  # Better quality than v1.5
    
    print(f"Loading model {model_id} on {device}...")
    
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None, 
        use_safetensors=True
    )
    

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
   
    pipe.enable_attention_slicing()
    
    pipe = pipe.to(device)
    
    return pipe

def generate_image(
    prompt: str,
    output_dir: str = "outputs",
    filename: Optional[str] = None,
    steps: int = 50,
    guidance_scale: float = 9.0,
    width: int = 768,
    height: int = 768,
    negative_prompt: Optional[str] = None,
    num_images: int = 1,
    seed: Optional[int] = None,
    init_image: Optional[Union[str, Image.Image]] = None,
    strength: float = 0.7,
    mask_image: Optional[Union[str, Image.Image]] = None,
    variation_strength: float = 0.0,
    batch_name: Optional[str] = None,
    **kwargs
) -> Union[str, List[str]]:
    """
    Generate one or more images using Stable Diffusion with advanced options.
    
    Args:
        prompt: Text prompt for image generation
        output_dir: Directory to save generated images
        filename: Output filename (will append _n for multiple images)
        steps: Number of denoising steps (20-100)
        guidance_scale: How closely to follow the prompt (1-20)
        width: Image width (must be multiple of 8)
        height: Image height (must be multiple of 8)
        negative_prompt: What to avoid in the generated image
        num_images: Number of images to generate
        seed: Random seed for reproducibility
        init_image: Path to image or PIL Image for img2img
        strength: Strength of img2img (0-1, lower preserves original more)
        mask_image: Mask for inpainting (black=keep, white=replace)
        variation_strength: How much to vary the image (0-1)
        batch_name: Name for this batch of images
        
    Returns:
        List of paths to generated images or single path if num_images=1
    """
    # Set up device and random seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device).manual_seed(seed)
    
    # Set up negative prompt
    if negative_prompt is None:
        negative_prompt = ("low quality, blurry, distorted, disfigured, text, watermark, "
                         "signature, lowres, bad anatomy, bad hands, error")
    
    # Ensure dimensions are multiples of 8
    width = max(64, (width // 8) * 8)
    height = max(64, (height // 8) * 8)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up batch directory if needed
    if batch_name:
        batch_dir = os.path.join(output_dir, batch_name)
        os.makedirs(batch_dir, exist_ok=True)
    else:
        batch_dir = output_dir
    
    # Handle image generation based on mode
    pipe = setup_model(device)
    
    # Set up generation parameters
    common_args = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'num_inference_steps': steps,
        'guidance_scale': guidance_scale,
        'width': width,
        'height': height,
        'generator': generator,
    }
    
    # Handle different generation modes
    if init_image is not None:
        # Convert init_image to PIL if it's a path
        if isinstance(init_image, str):
            init_image = Image.open(init_image).convert("RGB")
            init_image = init_image.resize((width, height))
            
        if mask_image is not None:
            # Inpainting mode
            if isinstance(mask_image, str):
                mask_image = Image.open(mask_image).convert("L")
                mask_image = mask_image.resize((width, height))
            
            pipe = StableDiffusionInpaintPipeline(**{
                k: v for k, v in pipe.components.items()
                if k in ['vae', 'text_encoder', 'tokenizer', 'unet', 'scheduler']
            })
            pipe = pipe.to(device)
            
            images = pipe(
                image=init_image,
                mask_image=mask_image,
                strength=strength,
                num_images_per_prompt=num_images,
                **common_args
            ).images
        else:
            # Img2Img mode
            pipe = StableDiffusionImg2ImgPipeline(**{
                k: v for k, v in pipe.components.items()
                if k in ['vae', 'text_encoder', 'tokenizer', 'unet', 'scheduler', 'safety_checker', 'feature_extractor']
            })
            pipe = pipe.to(device)
            
            images = pipe(
                image=init_image,
                strength=strength,
                num_images_per_prompt=num_images,
                **common_args
            ).images
    else:
        # Text2Img mode
        if variation_strength > 0:
            # Add variation to the prompt
            variation_prompt = f"{prompt}, variation: {variation_strength:.2f}"
            common_args['prompt'] = [prompt, variation_prompt]
            common_args['num_images_per_prompt'] = num_images // 2 + num_images % 2
        else:
            common_args['num_images_per_prompt'] = num_images
        
        images = pipe(**common_args).images
        
        # If we did variation, we need to flatten the list
        if variation_strength > 0 and len(images) > num_images:
            images = images[:num_images]
        
    # Save all generated images
    output_paths = []
    for i, image in enumerate(images):
        # Generate filename
        if filename:
            if num_images > 1:
                base, ext = os.path.splitext(filename)
                img_filename = f"{base}_{i+1}{ext}"
            else:
                img_filename = filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_filename = f"generated_{timestamp}_{i+1}.png"
        
        output_path = os.path.join(batch_dir, img_filename)
        
        # Post-processing
        if 'post_process' in kwargs and kwargs['post_process']:
            image = post_process_image(image)
        
        # Save image
        image.save(output_path)
        output_paths.append(output_path)
    
    print(f"✅ Generated {len(images)} image(s) in {batch_dir}")
    return output_paths[0] if len(output_paths) == 1 else output_paths

def post_process_image(image: Image.Image) -> Image.Image:
    """Apply post-processing to enhance the image quality."""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Slight sharpening
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2)
    
    # Slight contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    
    return image

def create_variations(
    image_path: str,
    output_dir: str = "outputs/variations",
    num_variations: int = 4,
    strength: float = 0.3,
    prompt: Optional[str] = None,
    **kwargs
) -> List[str]:
    """
    Create variations of an existing image.
    
    Args:
        image_path: Path to the source image
        output_dir: Directory to save variations
        num_variations: Number of variations to create
        strength: How much to vary the image (0-1)
        prompt: Optional prompt to guide variations
        
    Returns:
        List of paths to generated variations
    """
    # Load the source image
    init_image = Image.open(image_path).convert("RGB")
    
    # If no prompt is provided, create a generic one
    if prompt is None:
        prompt = "A variation of the image"
    
    # Generate variations using img2img with different seeds
    variation_paths = []
    for i in range(num_variations):
        variation = generate_image(
            prompt=prompt,
            init_image=init_image,
            strength=strength,
            output_dir=output_dir,
            filename=f"variation_{i+1}.png",
            seed=random.randint(0, 2**32 - 1),
            **kwargs
        )
        variation_paths.append(variation)
    
    return variation_paths

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate images using Stable Diffusion with advanced options')
    
    # Basic generation parameters
    parser.add_argument('prompt', type=str, nargs='?', default='A beautiful sunset over a mountain landscape',
                      help='Text prompt for image generation')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, default=None,
                      help='Output filename (default: generated_<timestamp>.png)')
    parser.add_argument('--output-dir', type=str, default="outputs",
                      help='Directory to save generated images (default: outputs/)')
    parser.add_argument('--batch-name', type=str, default=None,
                      help='Name for this batch of images (creates subdirectory)')
    
    # Image generation parameters
    parser.add_argument('--steps', type=int, default=50,
                      help='Number of inference steps (default: 50, higher=better quality but slower)')
    parser.add_argument('--guidance', type=float, default=9.0,
                      help='Guidance scale (default: 9.0, 7-11 is good range)')
    parser.add_argument('--width', type=int, default=768,
                      help='Image width (default: 768, must be multiple of 8)')
    parser.add_argument('--height', type=int, default=768,
                      help='Image height (default: 768, must be multiple of 8)')
    parser.add_argument('--negative-prompt', type=str, default=None,
                      help='Negative prompt to guide generation away from unwanted content')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    
    # Batch and variations
    parser.add_argument('--num-images', '-n', type=int, default=1,
                      help='Number of images to generate (default: 1)')
    parser.add_argument('--variation-strength', type=float, default=0.0,
                      help='Strength of variations between images (0-1, default: 0.0)')
    
    # Image-to-image options
    parser.add_argument('--init-image', type=str, default=None,
                      help='Path to initial image for img2img or inpainting')
    parser.add_argument('--mask-image', type=str, default=None,
                      help='Path to mask image for inpainting (black=keep, white=replace)')
    parser.add_argument('--strength', type=float, default=0.7,
                      help='Strength for img2img/inpainting (0-1, default: 0.7)')
    
    # Post-processing
    parser.add_argument('--post-process', action='store_true',
                      help='Apply post-processing to enhance image quality')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')
    
    # Variations command
    variations_parser = subparsers.add_parser('variations', help='Create variations of an existing image')
    variations_parser.add_argument('image_path', type=str, help='Path to source image')
    variations_parser.add_argument('--num-variations', type=int, default=4,
                                 help='Number of variations to create (default: 4)')
    variations_parser.add_argument('--variation-strength', type=float, default=0.3,
                                 help='Strength of variations (0-1, default: 0.3)')
    variations_parser.add_argument('--prompt', type=str, default=None,
                                 help='Optional prompt to guide variations')
    variations_parser.add_argument('--output-dir', type=str, default="outputs/variations",
                                 help='Directory to save variations (default: outputs/variations/)')
    
    args = parser.parse_args()
    
    print("🚀 Starting AI Image Generator")
    print("=" * 40)
    
    try:
        # Handle variations command
        if args.command == 'variations':
            print(f"🔄 Creating {args.num_variations} variations of {args.image_path}")
            output_paths = create_variations(
                image_path=args.image_path,
                output_dir=args.output_dir,
                num_variations=args.num_variations,
                strength=args.variation_strength,
                prompt=args.prompt,
                post_process=args.post_process
            )
            print(f"✅ Created {len(output_paths)} variations in {args.output_dir}")
            return 0
        
        # Handle standard image generation
        print(f"🎨 Generating {args.num_images} image(s) with prompt: '{args.prompt}'")
        
        # Prepare output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate images
        output_paths = generate_image(
            prompt=args.prompt,
            output_dir=args.output_dir,
            filename=args.output,
            steps=args.steps,
            guidance_scale=args.guidance,
            width=args.width,
            height=args.height,
            negative_prompt=args.negative_prompt,
            num_images=args.num_images,
            seed=args.seed,
            init_image=args.init_image,
            mask_image=args.mask_image,
            strength=args.strength,
            variation_strength=args.variation_strength,
            batch_name=args.batch_name,
            post_process=args.post_process
        )
        
        # Display info about generated images
        if isinstance(output_paths, str):
            output_paths = [output_paths]
            
        for i, path in enumerate(output_paths):
            try:
                image = Image.open(path)
                print(f"\n🖼️  Image {i+1}: {os.path.basename(path)}")
                print(f"   Size: {image.size[0]}x{image.size[1]} pixels")
                print(f"   File size: {os.path.getsize(path) / 1024:.1f} KB")
            except Exception as e:
                print(f"⚠️ Could not display image details for {path}: {str(e)}")
        
        print(f"\n✨ Generated {len(output_paths)} image(s) successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by the user.")
        return 1
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
        return 1

if __name__ == "__main__":
    exit(main())
