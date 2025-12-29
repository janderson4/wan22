# 1. Base Image - CUDA 12.4
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 2. System dependencies (including git and nsight-systems for profiling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ffmpeg libsm6 libxext6 libgl1 build-essential ninja-build \
    cuda-nsight-systems-12-4 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Build tools & SageAttention
RUN pip install --upgrade pip setuptools wheel packaging
# Added einops and moviepy here to fix runtime errors
RUN pip install einops moviepy
RUN TORCH_CUDA_ARCH_LIST="9.0" \
    pip install sageattention==1.0.6 --no-build-isolation

# 4. CLONE TURBODIFFUSION (keeping your original approach)
RUN git clone https://github.com/thu-ml/TurboDiffusion.git /TurboDiffusion_Lib && \
    cd /TurboDiffusion_Lib && \
    git submodule update --init --recursive

# 5. SETUP YOUR RUNPOD FILES
WORKDIR /app
COPY . .

# 6. Install dependencies
RUN if [ -f /TurboDiffusion_Lib/requirements.txt ]; then \
    pip install --no-cache-dir -r /TurboDiffusion_Lib/requirements.txt; fi

# Ensure einops is present even if not in requirements
RUN pip install einops

# 7. EXPOSE THE LIBRARY TO PYTHON
ENV PYTHONPATH="/TurboDiffusion_Lib:/app:${PYTHONPATH}"

# 8. Fetch Models - check both locations
RUN if [ -f builder/fetch_models.py ]; then \
    python3 builder/fetch_models.py; \
    elif [ -f /TurboDiffusion_Lib/builder/fetch_models.py ]; then \
    python3 /TurboDiffusion_Lib/builder/fetch_models.py; \
    fi

CMD ["python3", "-u", "rp_handler.py"]
