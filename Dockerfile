# Use a robust CUDA base image
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# 1. Set environment to non-interactive to avoid build hangs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 2. INSTALL SYSTEM TOOLS FIRST
# We must install git before pip tries to use it in step 16
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    build-essential \
    ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

# 3. CLONE THE REPO
RUN git clone https://github.com/thu-ml/TurboDiffusion.git
WORKDIR /TurboDiffusion

# 4. INSTALL PYTHON DEPENDENCIES
# Upgrade pip first to handle modern wheel builds
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now git is available, so this will work:
RUN pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation

# Install the local turbodiffusion package
RUN pip install -e . --no-build-isolation

# 5. FETCH MODELS & COPY HANDLER
COPY builder/fetch_models.py /TurboDiffusion/builder/
RUN python builder/fetch_models.py

COPY rp_handler.py /TurboDiffusion/

CMD ["python", "-u", "rp_handler.py"]
