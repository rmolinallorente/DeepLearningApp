
import torch  # Import PyTorch for handling computations
from transformers import CLIPTextModel, CLIPTokenizer  # Import CLIP text model and tokenizer for text encoding
from diffusers import StableDiffusionPipeline  # Import Stable Diffusion pipeline for generating images
import PIL.Image as Image  # Import PIL (Python Imaging Library) for image processing

# Check if a GPU is available and set the device to CUDA if it is, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion pipeline with the pre-trained model from Hugging Face
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

# Function to generate an image from a text prompt
def generate_image_from_prompt(prompt):
    """
    This function takes a text prompt as input and generates an image using the Stable Diffusion model.
    
    Args:
    - prompt (str): The text description to generate the image from.

    Returns:
    - PIL.Image: The generated image.
    """
    # Generate the image from the prompt without computing gradients (saves memory and computation)
    with torch.no_grad():
        image = pipe(prompt).images[0]  # Generate the image using the pipeline
    return image