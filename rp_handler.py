import os
import sys
import torch
import runpod
import base64
import requests
import uuid
import time
import subprocess
from PIL import Image
from io import BytesIO

# Add TurboDiffusion to path if not already there
if '/TurboDiffusion_Lib' not in sys.path:
    sys.path.insert(0, '/TurboDiffusion_Lib')

# Try to import - adjust based on actual structure
try:
    from turbodiffusion.inference.wan2_2_i2v_infer import Wan22I2VInfer
    print("✓ Successfully imported Wan22I2VInfer from turbodiffusion package")
except ImportError as e:
    print(f"Import attempt 1 failed: {e}")
    try:
        # Alternative import path
        from turbodiffusion.models.wan22_i2v import Wan22I2V_Infer as Wan22I2VInfer
        print("✓ Successfully imported from turbodiffusion.models")
    except ImportError as e2:
        print(f"Import attempt 2 failed: {e2}")
        # Last resort - direct file import
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "wan22_i2v", 
            "/TurboDiffusion_Lib/turbodiffusion/inference/wan2_2_i2v_infer.py"
        )
        wan22_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wan22_module)
        Wan22I2VInfer = wan22_module.Wan22I2VInfer
        print("✓ Successfully imported via direct file loading")

# Global variable to keep model in memory
MODEL = None

def download_image(url):
    """Download image from URL"""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def handler(job):
    global MODEL
    job_input = job['input']
    
    # 1. Extract inputs with defaults
    prompt = job_input.get('prompt')
    image_url = job_input.get('image_url')
    steps = job_input.get('steps', 4)
    seed = job_input.get('seed', 0)
    num_frames = job_input.get('num_frames', 81)
    fps = job_input.get('fps', 24)
    enable_profiling = job_input.get('enable_profiling', False)
    resolution = job_input.get('resolution', '720p')
    
    if not prompt or not image_url:
        return {"error": "Missing prompt or image_url"}
    
    print(f"\n{'='*60}")
    print(f"Processing request:")
    print(f"  Prompt: {prompt[:80]}...")
    print(f"  Steps: {steps}, Frames: {num_frames}, FPS: {fps}")
    print(f"  Resolution: {resolution}, Profiling: {enable_profiling}")
    print(f"{'='*60}\n")
    
    # 2. Load Model if not already loaded
    if MODEL is None:
        print("Loading TurboDiffusion model...")
        load_start = time.time()
        
        try:
            MODEL = Wan22I2VInfer(
                model="Wan2.2-A14B",
                vae_path="/app/checkpoints/Wan2.1_VAE.pth",
                text_encoder_path="/app/checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                high_noise_model_path="/app/checkpoints/TurboWan2.2-I2V-A14B-high-720P-quant.pth",
                low_noise_model_path="/app/checkpoints/TurboWan2.2-I2V-A14B-low-720P-quant.pth",
                quant_linear=True,
                attention_type="sageattention",  # Using sageattention as in your setup
                device="cuda"
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            return {"error": f"Failed to load model: {str(e)}"}
        
        load_time = time.time() - load_start
        print(f"✓ Model loaded in {load_time:.2f}s\n")
    
    # 3. Process Image
    print("Downloading input image...")
    try:
        input_img = download_image(image_url)
        print(f"✓ Image downloaded: {input_img.size}\n")
    except Exception as e:
        return {"error": f"Failed to download image: {str(e)}"}
    
    output_filename = f"/tmp/video_{uuid.uuid4()}.mp4"
    profile_filename = f"/tmp/profile_{uuid.uuid4()}.nsys-rep" if enable_profiling else None
    
    # 4. Generate with timing
    print(f"Generating video...")
    gen_start = time.time()
    
    try:
        if enable_profiling and os.path.exists("/usr/local/cuda/bin/nsys"):
            print("Running with Nsight Systems profiling...")
            # Save image temporarily for profiling script
            temp_img_path = f"/tmp/input_{uuid.uuid4()}.jpg"
            input_img.save(temp_img_path)
            
            # Create a minimal profiling script
            profile_script = f"/tmp/profile_script_{uuid.uuid4()}.py"
            with open(profile_script, "w") as f:
                f.write(f"""
import sys
sys.path.insert(0, '/TurboDiffusion_Lib')
from turbodiffusion.inference.wan2_2_i2v_infer import Wan22I2VInfer
from PIL import Image

model = Wan22I2VInfer(
    model="Wan2.2-A14B",
    vae_path="/app/checkpoints/Wan2.1_VAE.pth",
    text_encoder_path="/app/checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
    high_noise_model_path="/app/checkpoints/TurboWan2.2-I2V-A14B-high-720P-quant.pth",
    low_noise_model_path="/app/checkpoints/TurboWan2.2-I2V-A14B-low-720P-quant.pth",
    quant_linear=True,
    attention_type="sageattention",
    device="cuda"
)

img = Image.open('{temp_img_path}')
model.generate(
    prompt="{prompt.replace('"', '\\"')}",
    image=img,
    num_steps={steps},
    num_frames={num_frames},
    seed={seed},
    save_path="{output_filename}",
    resolution="{resolution}"
)
""")
            
            profile_cmd = [
                "/usr/local/cuda/bin/nsys", "profile",
                "--output", profile_filename,
                "--force-overwrite", "true",
                "python3", profile_script
            ]
            subprocess.run(profile_cmd, check=True)
            
            # Cleanup temp files
            os.remove(temp_img_path)
            os.remove(profile_script)
        else:
            # Normal generation without profiling
            MODEL.generate(
                prompt=prompt,
                image=input_img,
                num_steps=steps,
                num_frames=num_frames,
                seed=seed,
                save_path=output_filename,
                resolution=resolution
            )
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Generation failed: {str(e)}"}
    
    gen_time = time.time() - gen_start
    print(f"✓ Video generated in {gen_time:.2f}s\n")
    
    # 5. Check if output exists
    if not os.path.exists(output_filename):
        return {"error": "Video generation completed but output file not found"}
    
    # 6. Encode video to base64
    print("Encoding video...")
    with open(output_filename, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # 7. Encode profile report if enabled
    profile_base64 = None
    if enable_profiling and profile_filename and os.path.exists(profile_filename):
        print("Encoding profile report...")
        with open(profile_filename, "rb") as f:
            profile_base64 = base64.b64encode(f.read()).decode('utf-8')
        os.remove(profile_filename)
    
    # Cleanup
    if os.path.exists(output_filename):
        os.remove(output_filename)
    
    # Calculate video duration
    video_duration = num_frames / fps
    
    print(f"✓ Request completed successfully\n")
    
    result = {
        "video_base64": video_base64,
        "generation_time_seconds": round(gen_time, 2),
        "num_frames": num_frames,
        "fps": fps,
        "video_duration_seconds": round(video_duration, 2),
        "resolution": resolution,
        "num_steps": steps,
        "seed": seed
    }
    
    if profile_base64:
        result["profile_report_base64"] = profile_base64
        result["profile_filename"] = os.path.basename(profile_filename)
    
    return result

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("TurboDiffusion RunPod Handler Starting")
    print(f"{'='*60}\n")
    print(f"Python path: {sys.path}\n")
    runpod.serverless.start({"handler": handler})
