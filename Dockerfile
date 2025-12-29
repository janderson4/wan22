# Use the official PyTorch development image
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Force standard paths
ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    build-essential \
    ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Upgrade core python build tools
RUN pip install --upgrade pip setuptools wheel

# 3. Setup Working Directory
WORKDIR /TurboDiffusion

# 4. Clone TurboDiffusion contents into the current directory
# Note: Since RunPod pulls your repo, we only need to clone SpargeAttn manually
RUN git clone https://github.com/thu-ml/SpargeAttn.git /SpargeAttn

# 5. Install SpargeAttn from the local folder
# We do this BEFORE the main requirements to ensure build-time deps are met
RUN cd /SpargeAttn && pip install . --no-build-isolation

# 6. Now install the main requirements
# Make sure your requirements.txt does NOT contain the SpargeAttn git link
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Install the current TurboDiffusion package
# Assuming your Dockerfile is in the root of your repo
COPY . .
RUN pip install -e . --no-build-isolation

# 8. Fetch Models
# This stays the same
RUN python3 builder/fetch_models.py

CMD ["python3", "-u", "rp_handler.py"]
