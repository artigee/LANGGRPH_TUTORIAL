import uuid
from pathlib import Path

import requests
from decouple import config
from langchain.tools import tool
from openai import OpenAI
from pydantic import BaseModel, Field

IMAGE_DIRECTORY = Path(__file__).parent.parent / "images"
CLIENT = OpenAI(api_key=str(config("OPENAI_API_KEY")))

def image_downloader(image_url: str | None) -> str:
    """Downloads an image from a URL and saves it locally.
    
    Args:
        image_url: URL of the image to download, or None
    Returns:
        str: Path where image was saved, or error message if download failed
    """
    if not image_url:
        return "No image URL returned from API."
    response = requests.get(image_url)
    if response.status_code != 200:
        return "Could not download image from URL."
    
    unique_id: uuid.UUID = uuid.uuid4()
    image_path = IMAGE_DIRECTORY / f"{unique_id}.png"
    with open(image_path, "wb") as file:
        file.write(response.content)
    return str(image_path)

class GenerateImageInput(BaseModel):
    """Schema for the image generation input."""
    image_description: str = Field(
        description="A detailed description of the desired image."
    )

@tool("generate_image", args_schema=GenerateImageInput)
def generate_image(image_description: str) -> str:
    """Generate an image based on a detailed description.
    
    Uses DALL-E 3 to generate an image and saves it locally.
    
    Args:
        image_description: Detailed text description of the image to generate
    Returns:
        str: Path to the saved image file
    """
    response = CLIENT.images.generate(
        model="dall-e-3",          # Using DALL-E 3 model
        prompt=image_description,
        size="1024x1024",         # Standard square image size
        quality="standard",        # Standard quality (vs HD)
        n=1,                      # Generate one image
    )
    image_url = response.data[0].url
    return image_downloader(image_url)

if __name__ == "__main__":
    print(generate_image.run("A picture of sharks eating pizza in space."))

