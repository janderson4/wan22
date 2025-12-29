# We keep CUDA 12.4 for best performance on H100/A100/4090
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ffmpeg libsm6 libxext6 libgl1 build-essential ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Build tools - Added triton as it is a core requirement for sageattention
RUN pip install --upgrade pip setuptools wheel packaging triton

# 3. Install SageAttention
# Using version 1.0.6 as identified for the V1 implementation
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    pip install sageattention==1.0.6 --no-build-isolation

# 4. Setup TurboDiffusion
WORKDIR /TurboDiffusion
COPY . .

# Clean requirements.txt of the old problematic packages
RUN if [ -f requirements.txt ]; then \
    sed -i '/SpargeAttn/d' requirements.txt && \
    sed -i '/sageattn/d' requirements.txt && \
    sed -i '/sageattention/d' requirements.txt; \
    fi

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e . --no-build-isolation

# 5. Fetch Models
RUN python3 builder/fetch_models.py

CMD ["python3", "-u", "rp_handler.py"]
