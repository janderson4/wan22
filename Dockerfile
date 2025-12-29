# We keep CUDA 12.4 for best performance on H100/A100/4090
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ffmpeg libsm6 libxext6 libgl1 build-essential ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Build tools
RUN pip install --upgrade pip setuptools wheel packaging

# 3. Install SageAttn 
# Most research forks of TurboDiffusion now use this. 
# We install it directly via pip which handles the kernels better.
RUN pip install sageattn==1.0.1 --no-build-isolation

# 4. Setup TurboDiffusion
WORKDIR /TurboDiffusion
COPY . .

# Clean requirements.txt of the old problematic package
RUN if [ -f requirements.txt ]; then \
    sed -i '/SpargeAttn/d' requirements.txt; \
    fi

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e . --no-build-isolation

# 5. Fetch Models
RUN python3 builder/fetch_models.py

CMD ["python3", "-u", "rp_handler.py"]
