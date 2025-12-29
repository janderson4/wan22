FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. Set CUDA Paths (Crucial for SpargeAttn compilation)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 2. Install system dependencies + CUDA compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ffmpeg libsm6 libxext6 libgl1 build-essential ninja-build \
    cuda-nvcc-12-1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Build-time tools
RUN pip install --upgrade pip setuptools wheel setuptools_scm packaging ninja

# 4. Install SpargeAttn using your Strategy 2 logic
RUN git clone https://github.com/thu-ml/SpargeAttn.git /SpargeAttn && \
    cd /SpargeAttn && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1 \
    MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    FORCE_CUDA=1 \
    python3 -m pip install -vvv . --no-build-isolation

# 5. Setup TurboDiffusion
WORKDIR /TurboDiffusion
COPY . .

# Clean requirements.txt
RUN if [ -f requirements.txt ]; then sed -i '/SpargeAttn/d' requirements.txt; fi
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e . --no-build-isolation

# 6. Fetch Models
RUN python3 builder/fetch_models.py

CMD ["python3", "-u", "rp_handler.py"]
