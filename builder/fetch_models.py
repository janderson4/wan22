import os
import requests

def download_file(url, save_path):
    if os.path.exists(save_path):
        return
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

MODELS = {
    "VAE": "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth",
    "T5": "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth",
    # Switched to unquantized (removed -quant from URLs)
    "TurboHigh": "https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-high-720P.pth",
    "TurboLow": "https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-low-720P.pth"
}

os.makedirs("checkpoints", exist_ok=True)
for name, url in MODELS.items():
    path = os.path.join("checkpoints", url.split("/")[-1])
    download_file(url, path)
