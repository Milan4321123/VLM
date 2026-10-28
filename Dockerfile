# Default: portable CPU inference with supplied bounding boxes.
#   docker build -t vlm-fo1:cpu .
# Optional NVIDIA/UPN build:
#   docker build --target gpu -t vlm-fo1:gpu .
# Model checkpoints are downloaded only when inference needs them, never at build.

ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONPATH=/app \
    HF_HOME=/root/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    VLM_FO1_DEVICE=cuda \
    VLM_FO1_LLM_MODEL=Qwen/Qwen3.5-0.8B \
    VLM_FO1_ATTN_IMPL=sdpa \
    UPN_CKPT=/app/resources/upn_large.pth \
    UPN_AUTO_DOWNLOAD=0

# Python 3.10, CUDA compiler and tools for the optional UPN extension.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev build-essential ninja-build \
        git ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app
RUN python -m pip install pip==25.0.1 setuptools==75.8.0 wheel==0.45.1 \
    && python -m pip install torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124
COPY requirements.txt /app/requirements.txt
COPY requirements-gpu.txt /app/requirements-gpu.txt
COPY detect_tools/upn/requirements.txt /app/detect_tools/upn/requirements.txt
RUN python -m pip install -r requirements-gpu.txt -r detect_tools/upn/requirements.txt \
    && python -m pip check
COPY . /app

# Building does not need a GPU. Limit compiler parallelism to reduce build RAM.
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
ARG MAX_JOBS=2
RUN FORCE_CUDA=1 MAX_JOBS="${MAX_JOBS}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
        python -m pip install --no-build-isolation --no-deps ./detect_tools/upn/ops \
    && python -c "import torch, MultiScaleDeformableAttention" \
    && mkdir -p /app/resources /app/outputs \
    && chmod +x /app/docker/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["python", "demo/gradio_demo.py"]

# The last stage is the default and has no CUDA or custom UPN extension.
# Official CPU wheels exist for both Linux x86_64 and arm64.
FROM python:3.10-slim-bookworm AS cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONPATH=/app \
    HF_HOME=/root/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    VLM_FO1_DEVICE=cpu \
    VLM_FO1_LLM_MODEL=Qwen/Qwen3.5-0.8B \
    VLM_FO1_ATTN_IMPL=sdpa \
    UPN_CKPT=/app/resources/upn_large.pth \
    UPN_AUTO_DOWNLOAD=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN python -m pip install pip==25.0.1 setuptools==75.8.0 wheel==0.45.1 \
    && python -m pip install torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r requirements.txt && python -m pip check
COPY . /app
RUN mkdir -p /app/resources /app/outputs \
    && chmod +x /app/docker/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["python", "demo/gradio_demo.py"]
