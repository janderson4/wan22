# 1. Base Image - CUDA 12.4 for modern GPU support
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Ensures Python looks in the current directory for modules
ENV PYTHONPATH="/TurboDiffusion:${PYTHONPATH}"

# 2. System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ffmpeg libsm6 libxext6 libgl1 build-essential ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Build tools & SageAttention (V1)
RUN pip install --upgrade pip setuptools wheel packaging triton
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    pip install sageattention==1.0.6 --no-build-isolation

# 4. Setup TurboDiffusion
WORKDIR /TurboDiffusion
COPY . .

# Remove problematic packages from requirements
RUN if [ -f requirements.txt ]; then \
    sed -i '/SpargeAttn/d' requirements.txt && \
    sed -i '/sageattn/d' requirements.txt && \
    sed -i '/sageattention/d' requirements.txt; \
    fi

# Install general dependencies (transformers, runpod, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Fetch Models
RUN python3 builder/fetch_models.py

CMD ["python3", "-u", "rp_handler.py"]
