FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

WORKDIR /
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y git wget ffmpeg libsm6 libxext6 libgl1-mesa-glx build-essential

# Clone TurboDiffusion
RUN git clone https://github.com/thu-ml/TurboDiffusion.git
WORKDIR /TurboDiffusion

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation
RUN pip install -e . --no-build-isolation

# Fetch models during build
COPY builder/fetch_models.py /TurboDiffusion/builder/
RUN python builder/fetch_models.py

# Final setup
COPY rp_handler.py /TurboDiffusion/
CMD ["python", "-u", "rp_handler.py"]
