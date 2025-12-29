FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. Install system dependencies + symlink git for absolute certainty
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    build-essential \
    ninja-build && \
    ln -sf /usr/bin/git /usr/local/bin/git && \
    ln -sf /usr/bin/git /bin/git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Set environment variables that Python/Pip look for
ENV PATH="/usr/local/bin:/usr/bin:/bin:${PATH}"
ENV GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git

WORKDIR /

# 3. Clone SpargeAttn manually
RUN git clone https://github.com/thu-ml/SpargeAttn.git /SpargeAttn

# 4. Install build-time dependencies first
RUN pip install --upgrade pip setuptools wheel setuptools_scm

# 5. Install SpargeAttn with explicit PATH injection
# We also use --no-build-isolation to force it to use the environment we just set up
RUN cd /SpargeAttn && \
    PATH="/usr/bin:/usr/local/bin:$PATH" python3 -m pip install . --no-build-isolation

# 6. Setup TurboDiffusion (cloned by RunPod GitHub integration)
WORKDIR /TurboDiffusion
COPY requirements.txt .

# Ensure requirements.txt doesn't have the git link for SpargeAttn
RUN sed -i '/SpargeAttn/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# 7. Install TurboDiffusion
COPY . .
RUN PATH="/usr/bin:/usr/local/bin:$PATH" python3 -m pip install -e . --no-build-isolation

# 8. Fetch Models
RUN python3 builder/fetch_models.py

CMD ["python3", "-u", "rp_handler.py"]
