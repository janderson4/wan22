FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ffmpeg libsm6 libxext6 libgl1 build-essential ninja-build && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Pre-install Python build-essential tools (from your Option 1)
# Including 'packaging' and 'ninja' is crucial for CUDA packages
RUN pip install --upgrade pip setuptools wheel setuptools_scm packaging ninja

WORKDIR /

# 3. Clone SpargeAttn
RUN git clone https://github.com/thu-ml/SpargeAttn.git /SpargeAttn

# 4. INSTALL SPARGEATTN (Combining all strategies)
# - Verbose (-v) to see exactly what fails
# - Pretend Version to skip the Git check
# - No Build Isolation to use our pre-installed ninja/packaging
RUN cd /SpargeAttn && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1 \
    MAX_JOBS=4 \
    python3 -m pip install -v . --no-build-isolation

# 5. SETUP TURBODIFFUSION
WORKDIR /TurboDiffusion
COPY . .

# Remove SpargeAttn from requirements.txt if present
RUN if [ -f requirements.txt ]; then sed -i '/SpargeAttn/d' requirements.txt; fi

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e . --no-build-isolation

# 6. FETCH MODELS
RUN python3 builder/fetch_models.py

CMD ["python3", "-u", "rp_handler.py"]
