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

# Set PYTHONPATH
os.environ['PYTHONPATH'] = '/TurboDiffusion_Lib'
sys.path.insert(0, '/TurboDiffusion_Lib')

def download_image(url):
    """Download image from URL"""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def handler(job):
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
    
    # 2. Download image
    print("Downloading input image...")
    try:
        input_img = download_image(image_url)
        print(f"✓ Image downloaded: {input_img.size}\n")
    except Exception as e:
        return {"error": f"Failed to download image: {str(e)}"}
    
    # Save image temporarily
    temp_img_path = f"/tmp/input_{uuid.uuid4()}.jpg"
    input_img.save(temp_img_path)
    
    output_filename = f"/tmp/video_{uuid.uuid4()}.mp4"
    profile_filename = f"/tmp/profile_{uuid.uuid4()}.nsys-rep" if enable_profiling else None
    
    # 3. Generate using the official inference script
    print(f"Generating video...")
    gen_start = time.time()
    
    # Build command to run the official inference script
    cmd = [
        "python3", "/TurboDiffusion_Lib/turbodiffusion/inference/wan2.2_i2v_infer.py",
        "--model", "Wan2.2-A14B",
        "--low_noise_model_path", "/app/checkpoints/TurboWan2.2-I2V-A14B-low-720P-quant.pth",
        "--high_noise_model_path", "/app/checkpoints/TurboWan2.2-I2V-A14B-high-720P-quant.pth",
        "--vae_path", "/app/checkpoints/Wan2.1_VAE.pth",
        "--text_encoder_path", "/app/checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
        "--resolution", resolution,
        "--image_path", temp_img_path,
        "--prompt", prompt,
        "--num_samples", "1",
        "--num_steps", str(steps),
        "--num_frames", str(num_frames),
        "--seed", str(seed),
        "--save_path", output_filename,
        "--quant_linear",
        "--attention_type", "sageattention",
        "--ode"
    ]
    
    env = {**os.environ, 'PYTHONPATH': '/TurboDiffusion_Lib'}
    
    try:
        if enable_profiling and os.path.exists("/usr/local/cuda/bin/nsys"):
            print("Running with Nsight Systems profiling...")
            profile_cmd = [
                "/usr/local/cuda/bin/nsys", "profile",
                "--output", profile_filename,
                "--force-overwrite", "true"
            ] + cmd
            
            result = subprocess.run(
                profile_cmd, 
                check=True, 
                env=env,
                capture_output=True,
                text=True
            )
        else:
            # Normal generation
            result = subprocess.run(
                cmd, 
                check=True, 
                env=env,
                capture_output=True,
                text=True
            )
        
        # Print output for debugging
        if result.stdout:
            print("Script output:", result.stdout)
        if result.stderr:
            print("Script stderr:", result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Error during generation: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return {"error": f"Generation failed: {e.stderr}"}
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Generation failed: {str(e)}"}
    finally:
        # Cleanup temp image
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
    
    gen_time = time.time() - gen_start
    print(f"✓ Video generated in {gen_time:.2f}s\n")
    
    # 4. Check if output exists
    if not os.path.exists(output_filename):
        return {"error": "Video generation completed but output file not found"}
    
    # 5. Encode video to base64
    print("Encoding video...")
    with open(output_filename, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # 6. Encode profile report if enabled
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
    runpod.serverless.start({"handler": handler})
