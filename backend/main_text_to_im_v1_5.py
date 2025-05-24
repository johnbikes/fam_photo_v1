import os
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionPipeline
from typing import Optional
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Text-to-Image API", description="API for generating images from text using Stable Diffusion")

# Pydantic model for request body
class TextToImageRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 40
    guidance_scale: float = 7.5

class OffloadRequest(BaseModel):
    offload: bool = True

# Global variable to hold the model pipeline
model_pipeline: Optional[StableDiffusionPipeline] = None
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# TODO: could add on the refiner
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Load the Stable Diffusion model into memory."""
    global model_pipeline
    try:
        logger.info(f"Loading model {model_id} on {device}...")
        model_pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True, 
            variant="fp16",
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN")  # Optional: Set Hugging Face token in environment
        )
        if device == "cuda":
            model_pipeline = model_pipeline.to(device)
            model_pipeline.enable_attention_slicing()  # Optimize memory usage
        logger.info(f"Model loaded successfully to {device = }.")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

def offload_model():
    """Offload the model from GPU to free memory."""
    global model_pipeline
    try:
        if model_pipeline is not None:
            logger.info("Offloading model from memory...")
            if device == "cuda":
                model_pipeline.to("cpu")  # Move to CPU before deleting
                torch.cuda.empty_cache()  # Clear GPU cache
            model_pipeline = None
            logger.info("Model offloaded successfully.")
        else:
            logger.info("No model loaded to offload.")
    except Exception as e:
        logger.error(f"Failed to offload model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model offloading failed: {str(e)}") # HA - weird extra parens ...

@app.on_event("startup")
async def startup_event():
    """Load the model when the app starts."""
    load_model()

@app.on_event("shutdown")
async def shutdown_event():
    """Offload the model when the app shuts down."""
    offload_model()

@app.post("/generate-image")
async def generate_image(request: TextToImageRequest):
    """Generate an image from a text prompt using Stable Diffusion."""
    global model_pipeline
    try:
        if model_pipeline is None:
            logger.info("Model not loaded, loading now...")
            load_model()

        logger.info(f"Generating image for prompt: {request.prompt}")
        # Generate image
        with torch.no_grad():
            image = model_pipeline(
                prompt=request.prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale
            ).images[0]

        # Save image to a temporary file
        image_path = "generated_image.png"
        image.save(image_path)
        logger.info("Image generated successfully.")

        # Read image as bytes
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Clean up
        os.remove(image_path)

        return {"image": image_bytes.hex(), "message": "Image generated successfully"}
    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.post("/manage-model")
async def manage_model(request: OffloadRequest):
    """Manage model loading or offloading."""
    try:
        if request.offload:
            offload_model()
            return {"message": "Model offloaded successfully"}
        else:
            if model_pipeline is None:
                load_model()
            return {"message": "Model loaded successfully"}
    except Exception as e:
        logger.error(f"Model management failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model management failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Check the health of the API."""
    return {"status": "healthy", "device": device, "model_loaded": model_pipeline is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)