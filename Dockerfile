# Upgrading to 12.4 to satisfy the Compute Capability 8.9 requirement
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

# 3. Install SpargeAttn
# We keep the 'pretend version' just in case the Git error tries to return
RUN git clone https://github.com/thu-ml/SpargeAttn.git /SpargeAttn && \
    cd /SpargeAttn && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1 \
    MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    pip install . --no-build-isolation

# 4. Setup TurboDiffusion
WORKDIR /TurboDiffusion
COPY . .

# Clean requirements.txt
RUN if [ -f requirements.txt ]; then \
    sed -i '/SpargeAttn/d' requirements.txt; \
    fi

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e . --no-build-isolation

# 5. Fetch Models
RUN python3 builder/fetch_models.py

CMD ["python3", "-u", "rp_handler.py"]
