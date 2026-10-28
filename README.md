# VLM-FO1 with a Qwen3.5 decoder

Inference-only migration of VLM-FO1's language decoder to **`Qwen/Qwen3.5-0.8B`**, with a command-line interface, Gradio interface, and CPU/NVIDIA Docker configurations. **No fine-tuning is performed.**

The original VLM-FO1 vision encoders and Hybrid Fine-grained Region Encoder (HFRE) are retained. Their 2048-dimensional features pass through a new 1024-dimensional bridge into Qwen3.5. **The bridge and added region-token embeddings are untrained.** This is an architecture and deployment experiment; its answers must not be presented as validated object detection or counting results. See [migration details](docs/qwen35_migration.md).

Generation uses Transformers' official `Qwen3_5ForCausalLM` text decoder. Qwen3.5's native vision tower is omitted; the retained VLM-FO1 encoders and HFRE supply all visual features.

## Run on Google Colab (no local GPU needed)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Milan4321123/VLM/blob/qwen35-decoder-swap/notebooks/colab_demo.ipynb)

Select *Runtime → Change runtime type → T4 GPU*, then *Run all*. The notebook installs the pinned dependencies, runs the offline tests, runs the full pipeline on the fruit example, and can start the web demo with a public link.

## Start on the receiving computer

Install Docker Engine with Compose v2, or Docker Desktop, then extract this project and open a terminal **inside this directory**. The computer needs internet access for the initial image build and model download.

The commands below use Bash syntax. On Windows, use a WSL2 terminal with Docker Desktop's WSL integration and Linux containers enabled.

### CPU: works without an NVIDIA GPU

```bash
docker compose build
docker compose run --rm vlm-fo1 python cli.py --check
docker compose up
```

Open **http://localhost:8000**. The page opens without loading model weights. Use the included fruit example, supply boxes as JSON, or draw regions before submitting. Model weights load on the first inference request. Stop with Ctrl+C, then `docker compose down`.

For a reproducible command-line run with the included image and seven provided boxes:

```bash
docker compose run --rm vlm-fo1 python inference.py \
  --max-new-tokens 32 --output outputs/example.jpg --json outputs/example.json
```

The first inference downloads both the VLM-FO1 checkpoint (approximately **8.9 GB**) and Qwen3.5 (approximately **1.6 GB**). Budget **32 GB of system RAM and 30 GB free disk** for CPU use; these are conservative planning estimates, not measured minimum requirements. Give Docker enough RAM in its resource settings. CPU inference can take several minutes or longer, especially when the system must use disk as extra memory.

### NVIDIA GPU: optional Linux configuration

The host needs a compatible NVIDIA driver and NVIDIA Container Toolkit. The image uses CUDA 12.4 and PyTorch 2.6. The GPU image also compiles UPN's CUDA extension.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm vlm-fo1 \
  python inference.py --load-4bit --max-new-tokens 32 \
  --output outputs/example.jpg --json outputs/example.json
```

GPU memory needs depend on image size and generation length. Use 4-bit loading on constrained GPUs; the complete pipeline still includes two vision encoders. No GPU memory minimum or quantized accuracy is claimed as measured here.

For automatic UPN proposals, explicitly enable its checkpoint download:

```bash
UPN_AUTO_DOWNLOAD=1 docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm vlm-fo1 \
  python cli.py --image demo/demo_image.jpg --task od --target orange \
  --load-4bit --max-tokens 32 --json outputs/upn-example.json
```

UPN is optional and requires CUDA. Manual boxes work in both images. The default startup, `--help`, and `--check` never download weights. UPN is downloaded only when requested for actual inference. The UPN checkpoint is approximately 875 MB.

## Command-line interface

```bash
python cli.py --help
python cli.py --check
python cli.py --image demo/demo_image.jpg --task od --target orange \
  --boxes '[[161,11,292,127],[268,61,428,226]]' \
  --output outputs/orange.jpg --json outputs/orange.json
```

Inside Docker, prefix the Python command with `docker compose run --rm vlm-fo1`.

- `--prompt "..."` sends a custom prompt; `--task` fills a task template.
- Tasks: `od`, `count`, `ocr`, `brief-caption`, `detailed-caption`, `reason`, `grounding`, `all`.
- `--boxes` or `--boxes-file` accepts 1–100 `[x1,y1,x2,y2]` rectangles in original image pixels. Each box must have positive area and lie inside the image. Without either option, the CLI requests UPN proposals.
- `--model` selects the **VLM-FO1 vision checkpoint**. `--llm-model-path` selects the **generation decoder**, defaulting to Qwen3.5-0.8B. The Qwen2.5 text in the original checkpoint name does not mean that Qwen2.5 generates the answer.
- `--original-decoder` explicitly selects the original Qwen2.5 baseline.
- `--device auto` selects CUDA if available, otherwise CPU. `--load-4bit` and `--load-8bit` require the NVIDIA image.
- `--seed 0` and greedy generation are the defaults. Identical seeds do not guarantee identical outputs across different hardware and library implementations.
- `--max-tokens` limits generation (default 256). Start with 32 for deployment checks.

Standard output is one JSON document; diagnostics go to standard error. Results contain model IDs, device, proposal source and boxes, raw answer, parsed detections, seed, timing, and the untrained-bridge limitation. An empty detection list is allowed; it does not establish that the image contains no objects. The unmodified raw answer remains available for inspection.

`inference.py` is a shortcut using the included fruit image and its seven example boxes. For your own image, use the CLI with your own boxes. The Gradio interface and CLI use the same runtime and decoder defaults.

## Configuration and persisted files

| Setting | Default / meaning |
|---|---|
| `VLM_FO1_MODEL` | `omlab/VLM-FO1_Qwen2.5-VL-3B-v01` (Hub alias for `omlab/VLM-FO1-3B-v01`) |
| `VLM_FO1_LLM_MODEL` | `Qwen/Qwen3.5-0.8B` |
| `VLM_FO1_DEVICE` | Set by Compose: `cpu` or `cuda` |
| `VLM_FO1_ATTN_IMPL` | `sdpa`; no flash-attn installation required |
| `VLM_FO1_PORT` | Host interface port, default `8000` |
| `GRADIO_SHARE` | `0`; set to `1` for a public gradio.live link (Colab) |
| `UPN_AUTO_DOWNLOAD` | `0`; set to `1` to allow a UPN download on demand |
| `HF_TOKEN` | Optional Hugging Face token forwarded from the host shell |

`./outputs` contains generated images and JSON on the host. `./resources` stores UPN or manually supplied checkpoints. The named Docker volume `hf-cache` stores Hugging Face downloads. Normal `docker compose down` preserves that cache. Mount input directories explicitly, for example `docker compose run --rm -v /path/to/images:/data:ro vlm-fo1 python cli.py --image /data/photo.jpg ...`.

For offline inference, first populate the model cache on an online machine and transfer it, or place complete checkpoints under `resources/` and explicitly select both directories:

```bash
docker compose run --rm -e HF_HUB_OFFLINE=1 vlm-fo1 python inference.py \
  --model-path /app/resources/VLM-FO1-3B-v01 \
  --llm-model-path /app/resources/Qwen3.5-0.8B \
  --max-new-tokens 32 --json outputs/offline-example.json
```

Building the application image alone does not include the model weights. Never include a personal token in the submitted source archive.

## Verification

```bash
# Local environment: Python 3.10–3.12; no checkpoint downloads for unit tests
python -m venv .venv
# macOS/Linux:
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python cli.py --check

# Full receiving-machine Docker acceptance; saves evidence under outputs/
bash scripts/verify_container.sh cpu
# Or, on a configured NVIDIA host:
bash scripts/verify_container.sh gpu
```

[Validation notes](docs/validation.md) record the checks actually run for this submission and the checks still needed. Unit tests use small synthetic models to verify code behavior; they are not a model-quality benchmark. The separate `tests/smoke_qwen35_wrapper.py` runs real Qwen3.5 weights with a synthetic vision backbone and likewise does not validate grounding quality.

## Project and submission files

- [Project report](output/pdf/project_report.pdf), with editable [report text](docs/project_report.md).
- [Architecture and limitations](docs/qwen35_migration.md).
- [Detailed implementation reference](docs/implementation_reference.md): source map, data flow, settings, commands and evidence for reviewing or drafting a report.
- [Validation record](docs/validation.md).
- [Submission checklist](docs/submission.md).
- `cli.py`, `inference.py`, `demo/gradio_demo.py`: supported inference entry points.
- `vlm_fo1/`: model implementation and shared runtime.
- `detect_tools/upn/`: optional upstream proposal model and CUDA extension.
- `scripts/`: container verification, report generation, and submission packaging.

Unused upstream SAM3 demos, evaluation datasets, superseded scripts, old result
images and patch files have been moved to the local backup
`archive/legacy-files-2026-09-16.zip`. That backup is excluded from Docker and
the submission ZIP. The active project retains the model code, optional UPN,
four demo inputs, tests and verification evidence.

## Attribution

Based on [Om AI Lab's VLM-FO1](https://github.com/om-ai-lab/VLM-FO1), [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), and [ChatRex UPN](https://github.com/IDEA-Research/ChatRex). Existing dataset images and upstream source notices are retained. This migration does not claim the benchmark scores published by the original VLM-FO1 authors. The [historical README](docs/upstream_README.md) is retained for provenance, not as the submission runbook.
