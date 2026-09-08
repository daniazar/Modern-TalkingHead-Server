FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/hf_cache \
    TORCH_HOME=/app/torch_cache \
    TRANSFORMERS_CACHE=/app/hf_cache \
    DIFFUSERS_CACHE=/app/hf_cache \
    WEIGHTS_DIR=/app/weights \
    REPOS_DIR=/app/repos \
    PIP_PROGRESS_BAR=off \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0;PTX" \
    CUDA_MODULE_LOADING=LAZY \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.6" \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONPATH="/app:/app/repos:/app/vendor/Modern-TalkingHead-Server:/app/repos/AVTR-1:/app/repos/Ditto:/app/repos/FLOAT:/app/repos/FantasyTalking2:/app/repos/Hallo4:/app/repos/PersonaLive:/app/repos/SyncAnimation:/app/repos/EchoMimicV3:/app/repos/MuseTalk:${PYTHONPATH}"

WORKDIR /app

# 1. System utilities, multimedia codecs, and development libraries (cached layer)
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
        build-essential \
        curl \
        wget \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Python dependencies & ONNX Runtime GPU (cached layer)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --progress-bar off setuptools wheel && \
    (pip install --no-cache-dir --progress-bar off onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ || pip install --no-cache-dir --progress-bar off onnxruntime) && \
    pip install --no-cache-dir --progress-bar off --extra-index-url https://download.pytorch.org/whl/cu124 -r /app/requirements.txt || pip install --no-cache-dir --progress-bar off -r /app/requirements.txt


# 3. Dedicated directories for cached repos, model weights, and HuggingFace/Torch checkpoints (cached layer)
RUN mkdir -p /app/repos /app/weights /app/hf_cache /app/torch_cache /app/cache

# 4. Cache manager utility for upstream repos & models (cached layer)
COPY cache_manager.py /app/cache_manager.py

# 5. Application server code (COPIED LAST: changes to server.py/engine_loader.py build in < 1 second!)
COPY . /app/vendor/Modern-TalkingHead-Server

WORKDIR /app/vendor/Modern-TalkingHead-Server

EXPOSE 8010

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8010"]

