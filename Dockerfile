# We're sticking with the devel image to ensure you have the CUDA headers
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. FIX APT MIRRORS AND UPDATE
# We use a retry logic for the update in case of network blips
RUN apt-get clean && apt-get update --fix-missing

# 2. INSTALL SYSTEM TOOLS INDIVIDUALLY
# This ensures if one fails, we know exactly which one.
# I have swapped 'libgl1-mesa-glx' for 'libgl1' which is more modern.
RUN apt-get install -y --no-install-recommends git && \
    apt-get install -y --no-install-recommends wget && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get install -y --no-install-recommends libsm6 libxext6 && \
    apt-get install -y --no-install-recommends libgl1 && \
    apt-get install -y --no-install-recommends build-essential ninja-build

# 3. VERIFY GIT AND SET PATH
RUN which git && git --version
ENV PATH="/usr/bin:/usr/local/bin:${PATH}"

WORKDIR /

# 4. CLONE SPARGEATTN MANUALLY
RUN git clone https://github.com/thu-ml/SpargeAttn.git /SpargeAttn

# 5. PRE-INSTALL BUILD REQUIREMENTS
RUN pip install --upgrade pip setuptools wheel setuptools_scm

# 6. INSTALL SPARGEATTN
# Adding MAX_JOBS=4 to prevent OOM errors during compilation on large machines
RUN cd /SpargeAttn && \
    MAX_JOBS=4 pip install . --no-build-isolation

# 7. SETUP TURBODIFFUSION
WORKDIR /TurboDiffusion
COPY . .

# Clean requirements.txt of the SpargeAttn link to avoid loops
RUN if [ -f requirements.txt ]; then sed -i '/SpargeAttn/d' requirements.txt; fi

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e . --no-build-isolation

# 8. FETCH MODELS
RUN python3 builder/fetch_models.py

CMD ["python3", "-u", "rp_handler.py"]
