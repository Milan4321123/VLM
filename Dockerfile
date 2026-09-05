#
# VLM-FO1 inference image: VLM-FO1 (Qwen2.5-VL-3B) + UPN object proposals.
#
#   docker build -t vlm-fo1 .
#   docker run --rm --gpus all -p 8000:8000 \
#       -v "$PWD/resources:/app/resources" -v vlm-fo1-hf:/root/.cache/huggingface vlm-fo1
#
# See the "Run with Docker" section of README.md for the full usage.

ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONPATH=/app \
    HF_HOME=/root/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0

# Python 3.10 (Ubuntu 22.04 default), build tools for the CUDA op, libGL for OpenCV.
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        build-essential ninja-build git curl ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# 1) PyTorch. The PyPI build of torch 2.6.0 ships the CUDA 12.4 runtime libraries.
# hadolint ignore=DL3013
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install torch==2.6.0 torchvision==0.21.0

# 2) Project dependencies (VLM-FO1 + UPN).
COPY requirements.txt /app/requirements.txt
COPY detect_tools/upn/requirements.txt /app/detect_tools/upn/requirements.txt
RUN pip install -r /app/requirements.txt -r /app/detect_tools/upn/requirements.txt

# 3) flash-attn. The model is loaded with attn_implementation="flash_attention_2".
#    A prebuilt wheel is used because compiling flash-attn takes hours; the wheel
#    is picked to match the C++ ABI of the installed torch.
ARG FLASH_ATTN_VERSION=2.7.4.post1
RUN ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") \
    && pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.6cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl" \
    && python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"

# 4) Project source.
COPY . /app

# 5) Build the multi-scale deformable attention CUDA op used by UPN.
#    FORCE_CUDA=1 lets it compile on a build host without a GPU; the arch list
#    covers Volta through Hopper (trim it to your GPU to speed up the build).
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
WORKDIR /app/detect_tools/upn/ops
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
       pip install --no-build-isolation --no-deps . \
    && python -c "import torch, MultiScaleDeformableAttention; print('MultiScaleDeformableAttention OK')"
WORKDIR /app

RUN mkdir -p /app/resources /app/outputs && chmod +x /app/docker/entrypoint.sh

# Weights live outside the image: UPN checkpoint in /app/resources (downloaded by the
# entrypoint on first start) and the VLM-FO1 checkpoint in the Hugging Face cache.
VOLUME ["/app/resources", "/app/outputs", "/root/.cache/huggingface"]

EXPOSE 8000

ENTRYPOINT ["/app/docker/entrypoint.sh"]
# Default: the Gradio demo on http://localhost:8000. Override for one-shot inference:
#   docker run ... vlm-fo1 python cli.py --image demo/demo_image.jpg --prompt "..."
CMD ["python", "demo/gradio_demo.py"]
