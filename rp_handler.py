import os
import torch
import runpod
import base64
import requests
import uuid
from PIL import Image
from io import BytesIO

# Import TurboDiffusion internal logic (assuming repo is cloned into /TurboDiffusion)
import sys
sys.path.append('/TurboDiffusion')
from turbodiffusion.models.wan22_i2v import Wan22I2V_Infer # Example class name

# Global variable to keep model in memory
MODEL = None

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def handler(job):
    global MODEL
    job_input = job['input']
    
    # 1. Extract inputs
    prompt = job_input.get('prompt')
    image_url = job_input.get('image_url')
    steps = job_input.get('steps', 4)
    seed = job_input.get('seed', 0)

    if not prompt or not image_url:
        return {"error": "Missing prompt or image_url"}

    # 2. Load Model if not already loaded
    if MODEL is None:
        MODEL = Wan22I2V_Infer(
            model_type="Wan2.2-A14B",
            vae_path="checkpoints/Wan2.1_VAE.pth",
            text_encoder_path="checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
            high_noise_checkpoint="checkpoints/TurboWan2.2-I2V-A14B-high-720P-quant.pth",
            low_noise_checkpoint="checkpoints/TurboWan2.2-I2V-A14B-low-720P-quant.pth",
            quant_linear=True,
            device="cuda"
        )

    # 3. Process Image
    input_img = download_image(image_url)
    output_filename = f"/tmp/{uuid.uuid4()}.mp4"

    # 4. Generate
    MODEL.generate(
        prompt=prompt,
        image=input_img,
        num_steps=steps,
        seed=seed,
        save_path=output_filename,
        resolution="720p"
    )

    # 5. Encode video to base64
    with open(output_filename, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode('utf-8')

    # Cleanup
    if os.path.exists(output_filename):
        os.remove(output_filename)

    return {"video_base64": video_base64}

runpod.serverless.start({"handler": handler})
