> Cleanup note (16 September 2026): this is historical documentation. Optional SAM3/evaluation files, old results and Colab requirements were moved to `archive/legacy-files-2026-09-16.zip` in the local working copy and are excluded from the submission. Use the main [README](../README.md) for supported commands.

> Historical README from the starting branch. Commands and benchmark results below are not the submission instructions or Qwen3.5 results. Use the root README.

# VLM-FO1: Bridging the Gap Between High-Level Reasoning and Fine-Grained Perception

<p align="center">
 <a href="https://arxiv.org/pdf/2509.25916">
    <img 
        src="https://img.shields.io/badge/VLM_FO1-Paper-blue?logo=arxiv" 
    />
  </a> 
 <a href="https://huggingface.co/omlab/VLM-FO1-3B-v01">
    <img 
        src="https://img.shields.io/badge/VLM_FO1-Model-orange?logo=huggingface" 
    />
  </a>
 <a href="https://modelscope.cn/models/Om_AI_Lab/VLM-FO1-3B-v01">
    <img 
        src="https://img.shields.io/badge/ModelScope-%E6%A8%A1%E5%9E%8B-purple?logo=modelscope" 
    />
  </a>
 <a href="https://huggingface.co/spaces/P3ngLiu/VLM-FO1-3B-Demo">
    <img 
        src="https://img.shields.io/badge/VLM_FO1-Demo-orange?logo=huggingface" 
    />
  </a>
   <a href="https://huggingface.co/spaces/P3ngLiu/SAM3_VLM-FO1">
    <img 
        src="https://img.shields.io/badge/FO1+SAM3-Demo-orange?logo=huggingface" 
    />
  </a>
</p>

<p align="left">
VLM-FO1 endows pre-trained VLMs with superior fine-grained perception without compromising their inherent high-level reasoning and general understanding capabilities. It operates as a plug-and-play module that can be integrated with any existing VLM, establishing an effective and flexible paradigm for building the next generation of perception-aware models.
</p>

VLM-FO1 excels at a wide range of fine-grained perception tasks, including Object Grounding, Region Generative
Understanding, Visual Region Reasoning, and more.

For more details, check out our **[paper](https://arxiv.org/abs/2509.25916)**


## 📰 Update News

- **2026-03-12** — We are excited to announce that our models are now available on [ModelScope](https://modelscope.cn/models/Om_AI_Lab/VLM-FO1-3B-v01)!
- **2025-11-28** — We released a new [video tracking demo](#6-inference-with-sam3--vlm-fo1-video-tracking-gradio-demo) that combines **SAM3 video tracking** with **VLM-FO1**, enabling interactive object detection and tracking across video frames with complex natural language prompts.
- **2025-11-21** — We released a new demo that integrates **[SAM3](https://github.com/facebookresearch/sam3)** with **VLM-FO1**, enabling stronger segmentation fidelity and more reliable detection under complex, compositional label prompts. Check this section [Inference with SAM3 + VLM-FO1 Gradio Demo](#5-inference-with-sam3--vlm-fo1-gradio-demo) about how to run it. Or try the [huggingface demo](https://huggingface.co/spaces/P3ngLiu/SAM3_VLM-FO1)



<p align="center">
  <img src="https://raw.githubusercontent.com/om-ai-lab/VLM-FO1/main/assets/overall_results.png" alt="VLM-FO1 Overall Results" style="max-width:100%;">
</p>

## ✨ Key Features

🧩 **Plug-and-Play Modularity:** Our framework is designed as a set of enhancement modules that can be seamlessly integrated with any pre-trained VLM, preserving its original weights and capabilities.

🧠 **Hybrid Region Encoder (HFRE):** We introduce a novel Dual-Vision Encoder architecture that fuses semantic-rich features with perception-enhanced features, creating powerful region tokens that capture both high-level meaning and fine-grained spatial detail.

🎯 **State-of-the-Art Performance:** VLM-FO1 achieves SOTA results across a diverse suite of benchmarks.

✅ **Preserves General Abilities:** Our two-stage training strategy ensures that fine-grained perception is gained without causing catastrophic forgetting of the base model's powerful general visual understanding abilities.

<p align="center">
  <img src="../assets/structure.png" alt="VLM-FO1 Overall Results" style="max-width:100%;">
</p>



## 📝 ToDo
- [X] Release Gradio demo

## 🔗 Related works
If you are interested in our research, we welcome you to explore our other wonderful projects.
- 🔆 [Om AI Lab Blogs](https://om-ai-lab.github.io/index.html)
- 🔆 [VLM-R1: A stable and generalizable R1-style Large Vision-Language Model](https://github.com/om-ai-lab/VLM-R1/tree/main)
- 🔆 [OmAgent: Build Multimodal Language Agents with Ease](https://github.com/om-ai-lab/OmAgent)
- 🔆 [OmDet-Turbo: Real-time and accurate open-vocabulary end-to-end object detection](https://github.com/om-ai-lab/OmDet)
- 🔆 [ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration](https://github.com/om-ai-lab/ZoomEye)


## 📋 Table of Contents

- Installation
- Run with Docker
- Quick Start
  - Inference with Provided Bounding Boxes
  - Inference with Object Detector
  - Inference with Gradio Demo
  - Inference with SAM3 + VLM-FO1 Gradio Demo
  - Inference with SAM3 + VLM-FO1 Video Tracking Gradio Demo
  - Command-line Inference with a Custom Prompt
- Task Templates
- Evaluation
- Citation
- Acknowledgements

## 🛠️ Installation

### Requirements

- Python 3.10+
- PyTorch (GPU recommended). Please install the CUDA-enabled build that matches your system.
- Linux is the primary tested platform.

### Setup

```bash
git clone https://github.com/om-ai-lab/VLM-FO1.git
cd VLM-FO1
pip install -r requirements.txt
```

If you use conda:

```bash
conda create -n vlm-fo1 python=3.10 -y
conda activate vlm-fo1
pip install -r requirements.txt
```

### Google Colab (free GPU)

1. In Colab, select **Runtime > Change runtime type > T4 GPU**.
2. Run the following cells:

```python
!git clone --depth 1 --branch qwen35-decoder-swap https://github.com/Milan4321123/VLM.git /content/VLM
%cd /content/VLM
!pip install -q -r requirements-colab.txt
```

Restart the runtime after installation:

```python
import os
os.kill(os.getpid(), 9)
```

Then run 4-bit inference:

```python
%cd /content/VLM
!python inference.py --load-4bit --max-new-tokens 256
```

Display the generated visualization:

```python
from PIL import Image
from IPython.display import display
display(Image.open("/content/VLM/demo/vlm_fo1_result.jpg"))
```

The first run downloads the VLM-FO1 and Qwen3.5 checkpoints. Even with `--load-4bit`, large images or long generations can push past the T4's 16 GB. The `--load-4bit`/`--load-8bit` flags need `bitsandbytes`, which `requirements-colab.txt` already includes; for a local GPU setup install it separately. The current 2048-to-1024 Qwen3.5 bridge and newly added region-token embeddings require training for reliable grounding quality — see `docs/qwen35_migration.md`.
Additional notes:

- The model is loaded with `attn_implementation="flash_attention_2"`, so install [flash-attn](https://github.com/Dao-AILab/flash-attention) as well (a prebuilt wheel matching torch 2.6 / CUDA 12 / your Python version from the flash-attention releases page installs in seconds). If `flash_attn` is missing, the model falls back to `sdpa` and prints a warning; `VLM_FO1_ATTN_IMPL=flash_attention_2|sdpa|eager` overrides the choice.
- The Gradio demo and `cli.py` use UPN for object proposals, which needs the CUDA op built in `detect_tools/upn/ops` (see [Inference with UPN](#option-a-inference-with-upn)).
- Run the scripts from the repository root with the root on the module path, e.g. `PYTHONPATH=. python demo/gradio_demo.py`.
- If you do not want to set any of this up by hand, use the Docker image below.

## 🐳 Run with Docker

The repository ships a `Dockerfile`, a `docker-compose.yml` and a prompt-driven command-line script (`cli.py`), so the complete pipeline (UPN proposals + VLM-FO1) runs in a container without any manual environment setup.

### Prerequisites

- An NVIDIA GPU with roughly 16 GB of memory. The bundled flash-attn wheel needs an Ampere or newer GPU (RTX 30xx/40xx, A10, A100, L4, H100, ...); on older GPUs start the container with `VLM_FO1_ATTN_IMPL=sdpa`.
- NVIDIA driver 550 or newer (CUDA 12.4), Docker 24+ with Compose v2, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Around 30 GB of free disk for the image plus the checkpoints.

### 1) Build the image

```bash
docker compose build
# or without Compose:
docker build -t vlm-fo1 .
```

The build installs PyTorch 2.6 (CUDA 12.4), the project requirements, a prebuilt flash-attn wheel and compiles the UPN deformable-attention CUDA op. No GPU is needed at build time. To shorten the build, restrict the CUDA architectures to your GPU, e.g. `docker build --build-arg TORCH_CUDA_ARCH_LIST="8.6" -t vlm-fo1 .`.

### 2) Model weights

Weights are not baked into the image:

- The UPN checkpoint is downloaded automatically into `./resources/upn_large.pth` (a host folder mounted into the container) the first time the container starts.
- The VLM-FO1 checkpoint is fetched from the Hugging Face Hub on first use and cached in the `hf-cache` Docker volume. To use a local copy instead, place it under `./resources/VLM-FO1_Qwen2.5-VL-3B-v01` or set `VLM_FO1_MODEL` to another path or Hub id. Export `HF_TOKEN` in your shell if your Hub account requires it.

### 3) Run inference with your own prompt

```bash
docker compose run --rm vlm-fo1 python cli.py \
    --image demo/demo_image.jpg \
    --prompt "Please detect orange in this image. Answer the question with object indexes." \
    --output outputs/orange.jpg --json outputs/orange.json
```

The command prints a JSON document with the exact prompt that was used, the raw model answer and the selected boxes, and writes the annotated image to `./outputs/` on the host. Instead of a verbatim `--prompt`, a task template can be filled in with `--task` and `--target`:

```bash
docker compose run --rm vlm-fo1 python cli.py \
    --image demo/demo_image_01.jpg --task count --target "airplane with only one propeller"
```

Available tasks: `od`, `count`, `ocr`, `brief-caption`, `detailed-caption`, `reason`, `grounding`, `all` (see `python cli.py --help`). To run on your own pictures, mount a folder and refer to it inside the container:

```bash
docker compose run --rm -v /path/to/images:/data vlm-fo1 python cli.py \
    --image /data/photo.jpg --prompt "Please detect person, dog in this image. Answer the question with object indexes."
```

Pass `--boxes '[[x1,y1,x2,y2], ...]'` to use your own region proposals instead of UPN.

Without Compose:

```bash
docker run --rm --gpus all \
    -v "$PWD/resources:/app/resources" -v "$PWD/outputs:/app/outputs" \
    -v vlm-fo1-hf:/root/.cache/huggingface \
    vlm-fo1 python cli.py --image demo/demo_image.jpg --prompt "Please detect orange in this image. Answer the question with object indexes."
```

### 4) Run the Gradio demo

```bash
docker compose up
```

Then open http://localhost:8000. The same image can be started with `docker run --rm --gpus all -p 8000:8000 -v "$PWD/resources:/app/resources" -v vlm-fo1-hf:/root/.cache/huggingface vlm-fo1`.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `VLM_FO1_MODEL` | `omlab/VLM-FO1_Qwen2.5-VL-3B-v01` | VLM-FO1 checkpoint: Hub id or path inside the container |
| `UPN_CKPT` | `/app/resources/upn_large.pth` | UPN checkpoint path (downloaded if missing) |
| `SKIP_UPN_DOWNLOAD` | `0` | Set to `1` to skip the automatic UPN download |
| `VLM_FO1_ATTN_IMPL` | auto (`flash_attention_2` if installed, else `sdpa`) | Attention backend |
| `HF_TOKEN` | unset | Hugging Face token, forwarded from your shell when set |
| `PORT` | `8000` | Port of the Gradio demo inside the container |

The SAM3 demos (`demo/gradio_demo_with_sam3*.py`, `scripts/inference_with_sam3.py`) are not part of the image because SAM3 must be installed from its own repository with gated checkpoints; extend the Dockerfile if you need them.

## 🚀 Quick Start

### 1) Download Model Checkpoints

Download the pre-trained VLM-FO1 checkpoints from Hugging Face and place them under `resources/`:

- Pre-trained model: [Hugging Face link](https://huggingface.co/omlab/VLM-FO1-3B-v01)

### 2) Inference with Provided Bounding Boxes

Use `inference.py` to run VLM-FO1 with provided bounding boxes. 

Run:

```bash
python inference.py
```

The visualization with predicted boxes will be saved to `demo/vlm_fo1_result.jpg`.

### 3) Inference with Object Detector

#### Option A: Inference with UPN

**Note:** Due to company policy, we are unable to release the object detector OPN referenced in our paper. Instead, we provide integration with UPN (from [ChatRex](https://github.com/IDEA-Research/ChatRex)), which offers similar functionality. You are also free to use any object detector of your choice by preparing bounding box proposals compatible with our pipeline.

Before running, you need to install UPN's requirements and build the ops extension:

```bash
cd detect_tools/upn
pip install -r requirements.txt
cd ops
pip install -v -e .
```

Then, download the UPN checkpoint from [this link](https://github.com/IDEA-Research/ChatRex/releases/download/upn-large/upn_large.pth) and place it in your `resources/` folder:

```
wget https://github.com/IDEA-Research/ChatRex/releases/download/upn-large/upn_large.pth -P resources/
```

Use `scripts/inference_with_upn.py` to run FO1 with UPN-generated proposals.

Run:

```bash
python scripts/inference_with_upn.py
```

The script will:

- Load UPN and produce fine-grained object proposals
- Filter proposals by score
- Pass top proposals to FO1 for reasoning
- Save the visualization to `demo/vlm_fo1_result.jpg`

#### Option B: Inference with SAM3

We also provide integration with **[SAM3](https://github.com/facebookresearch/sam3)**, which offers high-quality open-vocabulary segmentation. This integration leverages SAM3 to generate object proposals based on text prompts, which are then refined and selected by VLM-FO1.

1.  Follow the official [SAM3 setup guide](https://github.com/facebookresearch/sam3) to install `sam3` and request access to the checkpoints.
2.  Download the SAM3 checkpoint (e.g., `sam3.pt`) and place it in `resources/sam3/`.

Use `scripts/inference_with_sam3.py` to run FO1 with SAM3-generated proposals.

Run:

```bash
python scripts/inference_with_sam3.py
```

The script will:

- Load SAM3 and generate object proposals based on a text prompt (e.g., "the ball nearest to the bear")
- Pass top proposals (bounding boxes) to FO1 for reasoning and filtering
- Save the visualization to `demo/vlm_fo1_result_with_sam3.jpg`

### 4) Inference with Gradio Demo
Run:

```bash
python demo/gradio_demo.py
```

### 5) Inference with SAM3 + VLM-FO1 Gradio Demo

1. Follow the official [SAM3 setup guide](https://github.com/facebookresearch/sam3) to install sam3 and request access to the checkpoints from Hugging Face.
2. Launch the joint demo:

```bash
python demo/gradio_demo_with_sam3.py
```

This script pipes SAM3’s text-conditioned segmentation into VLM-FO1’s complex label reasoning for more reliable detection and segmentation on compositional prompts.

### 6) Inference with SAM3 + VLM-FO1 Video Tracking Gradio Demo

Launch the video tracking demo:

```bash
python demo/gradio_demo_with_sam3_video_tracking.py
```

This demo extends the capabilities to video:
1.  **Frame Selection**: Pick a keyframe from your video.
2.  **Detection**: Use VLM-FO1 + SAM3 to identify objects with complex text prompts on the selected frame.
3.  **Tracking**: Propagate the segmentation masks forward through the video using SAM3's tracker.


### 7) Command-line Inference with a Custom Prompt

`cli.py` runs the full pipeline (UPN proposals + VLM-FO1) on one image with any prompt and prints the result as JSON:

```bash
python cli.py --image demo/demo_image.jpg \
    --prompt "Please detect orange in this image. Answer the question with object indexes."

python cli.py --image demo/demo_image_01.jpg --task count --target "airplane with only one propeller"
```

The annotated image is written to `outputs/<image name>_result.jpg` (change with `--output`), `--json` also saves the JSON result, and `--boxes` supplies your own proposals instead of UPN. See `python cli.py --help` for all options and the [Docker section](#-run-with-docker) for running it in a container.

## 📝 Task Templates

We provide task templates to facilitate tasks beyond detection (e.g., counting, grounding, and other vision-language reasoning tasks). See `vlm_fo1/task_templates` for examples. You can adapt the prompts or message construction to your specific application.

## Evaluation
**Model performance when integrated with UPN is as follows:**
| Model                          | COCO mAP | CountBench Accuracy | Pixmo-Count Accuracy |  HumanRef DF1/P/R |   LVIS SS/S-IoU | PACO SS/S-IoU | COCOText |
|---------------------------------|----------|----------|----------|----------|------|------|------|       
| VLM-FO1-3B-v01 (FO1 + UPN) |   44.4   |       89.73%        |       85.07%         |    82.6/86.8/83.5|   92.5/87   |  88.1/77.9 |  57.1 |

For benchmarking FO1 with UPN on your datasets, you can adapt the inference pipeline to export predictions and compare against ground-truth. We also include evaluation utilities under `evaluation/` for specific tasks.

- CountBench/Pixmo-Count evaluation: `evaluation/eval_countbench.py`
- COCO evaluation: `evaluation/eval_coco.py` (this script will output a prediction json file which you can use with the standard COCO evaluation tool to obtain scores)

## 📄 Citation

If you find VLM-FO1 useful in your research or applications, please cite our work:

```bibtex
@article{liu2025vlm,
  title={VLM-FO1: Bridging the Gap Between High-Level Reasoning and Fine-Grained Perception in VLMs},
  author={Liu, Peng and Shen, Haozhan and Fang, Chunxin and Sun, Zhicheng and Liao, Jiajia and Zhao, Tiancheng},
  journal={arXiv preprint arXiv:2509.25916},
  year={2025}
}
```

## 🙏 Acknowledgements
- Dataset images used above are from public benchmarks ([Pixmo-Count](https://huggingface.co/datasets/allenai/pixmo-count), [CountBenchQA](https://huggingface.co/datasets/vikhyatk/CountBenchQA)), credited to their respective authors.
- Thanks to the numerous researchers and developers who have contributed to the datasets used in our training.

