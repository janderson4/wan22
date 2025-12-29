# Using a slightly different base to ensure package manager stability
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. Manually set PATH to include standard bin directories
ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"

# 2. Separate APT updates and installs for better reliability
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    build-essential \
    ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. VERIFICATION STEP (This will force the build to fail early if git is missing)
RUN which git && git --version

WORKDIR /

# 4. CLONE TURBODIFFUSION
RUN git clone https://github.com/thu-ml/TurboDiffusion.git
WORKDIR /TurboDiffusion

# 5. INSTALL PYTHON DEPENDENCIES
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. INSTALL SPARGEATTN 
# We use a safer syntax here
RUN git config --global --add safe.directory /TurboDiffusion && \
    pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation

# 7. INSTALL TURBODIFFUSION
RUN pip install -e . --no-build-isolation

# 8. FETCH MODELS & HANDLER
COPY builder/fetch_models.py /TurboDiffusion/builder/
RUN python builder/fetch_models.py
COPY rp_handler.py /TurboDiffusion/

CMD ["python", "-u", "rp_handler.py"]
