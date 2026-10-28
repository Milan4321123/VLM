# Validation record

Prepared on **15 September 2026**. These checks establish software execution;
they do not establish detection, counting, or grounding accuracy.

## Tested environment

- Host: Apple Silicon Mac, 8 GB physical RAM.
- CPU container: Linux arm64, Python 3.10.21, PyTorch 2.6.0+cpu,
  torchvision 0.21.0, Transformers 5.2.0, Gradio 5.49.1.
- Isolated Linux VM: two CPUs; 4 GiB configured RAM and 6 GiB swap for
  complete pretrained inference. This was a constrained verification setup,
  not a recommended deployment specification.
- Tested runtime image: `vlm-fo1:cpu`,
  `sha256:efbc8e816027bcc6469b4fe6589b8e93f50c05afe496c395d94bb42b8058971e`.
  This identifies the 15 September build. The 16 September cleanup below
  removed unused files and dead code; it was verified separately with local tests.

## Completed checks

| Check | Result |
|---|---|
| Actual CPU Docker image build | Passed, including dependency consistency check |
| CPU and NVIDIA Compose configuration | Both parsed successfully; also checked after archive extraction |
| Offline regression suite inside the CPU image | **47 passed** in 34.95 seconds; one upstream timm deprecation warning |
| Container preflight | Passed; device `cpu`, intended Qwen3.5 decoder; no weights downloaded |
| Full pretrained Docker pipeline, one-token cap | Passed; real vision/HFRE and Qwen3.5; JSON and readable image saved |
| Full pretrained native CPU pipeline, 16-token cap | Passed on macOS arm64 with the Docker VM stopped; JSON and readable image saved |
| Browser startup | Page served on port 8000; Qwen3.5 controls, fruit image and seven manual boxes verified |
| Real Qwen3.5 wrapper smoke test | Passed with cached pretrained decoder and synthetic vision features |
| Archive handover | Required Docker build files present; ZIP integrity, extraction, help commands and shell syntax checked |

The suite covers malformed and out-of-bounds boxes, lazy startup, JSON output,
image-writing failures, example initialization, checkpoint structure and
loading, generation with the real tiny Qwen3.5 implementation, stopping
criteria, attention masks, reproducible frozen bridge initialization, the
optional original decoder's cache, and archive contents. The text-only loader
tests verify exact language-weight remapping from a multimodal checkpoint,
tied embeddings, identical cached generation scores with left padding, and
rejection of missing decoder weights.

The checkpoint regression deliberately fills non-persistent rotary buffers
with NaNs before their initialization. It verifies that loading restores the
correct frequencies, preserves saved weights and produces finite real vision
features. A separate test verifies rejection of non-finite visual features.

## Full-pipeline Docker check with a one-token limit

The final run uses the real retained VLM-FO1 vision encoders, HFRE and
projectors, followed by the real Qwen3.5-0.8B decoder. It uses the bundled fruit
image, seven supplied proposals, seed 0, greedy generation and a limit of one
new token. Model downloads were completed beforehand; this run is offline.
This constrained check is separate from the receiving-machine script, which
uses a 16-token limit.

Checkpoint revisions used:

- VLM-FO1: `15788f0d303509abacba5f97b52c9ff625080319`.
- Qwen3.5-0.8B: `2fc06364715b967f1860aea9cf38778875588b17`.

The recorded container invocation used staged local weights, which are
excluded from the source archive:

```bash
docker compose run --rm -T \
  -e HF_HUB_OFFLINE=1 -e GRADIO_ANALYTICS_ENABLED=False -e OMP_NUM_THREADS=1 \
  -v "$PWD/tmp/hf-container-cache:/root/.cache/huggingface:ro" \
  vlm-fo1 python inference.py --model-path resources/VLM-FO1-3B-v01 \
  --max-new-tokens 1 --output outputs/cpu-acceptance.jpg \
  --json outputs/cpu-acceptance.json
```

The process exited successfully. Inference took **357.167 seconds**;
the reported total request time was **537.366 seconds**, including checkpoint loading.
Image encoding took 122.2 seconds, region encoding 135.6 seconds and generation
96.0 seconds. The raw response was `Based`, truncated by the one-token limit,
and no region selection was parsed. The 500 x 399 output image was verified
readable. This checks complete model execution and file writing; it does not
constitute a complete answer to the prompt or a grounding-quality result.

An earlier test with 6 GiB VM RAM, 4 GiB swap and a 32-token limit was stopped
because of severe slowdown on this 8 GB host. Read-only profiling confirmed
that vision/region encoding had finished and generation advanced to 10 tokens;
that interrupted run is not counted as successful acceptance. A subsequent
4 GiB VM run with a 16-token limit was also stopped during generation.
Small Linux arithmetic benchmarks did not justify changing model arithmetic.
The successful one-token run uses the text-only Qwen loader, leaves more RAM
available to macOS and uses one inference thread. CLI diagnostics show image,
region and generation stages with elapsed times.

## Native CPU check with a 16-token limit

After the Docker VM was shut down, the same final inference implementation ran
directly on the Mac with Python 3.12.3, PyTorch 2.6.0, torchvision 0.21.0,
Transformers 5.2.0 and Gradio 5.49.1. It used the same cached checkpoint
revisions, image, seven proposals and seed, with two CPU threads:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
MPLCONFIGDIR=/private/tmp/fo1-mpl OMP_NUM_THREADS=2 GRADIO_ANALYTICS_ENABLED=False \
  .venv/bin/python -u inference.py --model-path resources/VLM-FO1-3B-v01 \
  --device cpu --max-new-tokens 16 --output outputs/native-acceptance.jpg \
  --json outputs/native-acceptance.json
```

The process exited successfully. Inference took **386.428 seconds**;
the reported total request time including loading was **418.895 seconds**.
Image encoding took 44.8 seconds, region encoding 45.6 seconds and generation
294.8 seconds. The response began `Based on the visual analysis of the image provided:`
and ended partway through a word because of the token cap. No region selection
was parsed. The saved JSON and 500 x 399 image passed the same metadata,
proposal and image-readability checks as the Docker result.

This is native macOS execution, not a completed 16-token Docker acceptance run.
Both outputs are truncated execution samples, not complete answers or measured
grounding results. Timings on this memory-constrained host are not a deployment
benchmark.

## Cleanup verification: 16 September 2026

Moved 26 unused files into a verified local backup outside the submission:
SAM3 demos and media, evaluation utilities and datasets, superseded scripts,
old result images, Colab requirements, and the two original patch files.
Each archived file was checked against its SHA256 before removing its original.
The cleanup also removed unused imports, constants and unreferenced helpers;
the supported model, UPN, demo inputs and inference behavior were preserved.
Redundant temporary checkpoint copies and extraction directories were removed,
freeing approximately 1.74 GiB; the original model weights remain available.

After cleanup, the full offline suite passed **47 tests in 9.64 seconds** on
the native environment above, with one upstream timm warning. The no-download
preflight passed. Evidence is in `cleanup-tests.txt` and
`cleanup-preflight.json` under `docs/evidence/`. Full pretrained runs and
Docker image builds were not repeated for this file/dead-code cleanup; the
15 September measurements above remain the recorded execution evidence.

## Final submission review: 17 September 2026

The code review found and fixed two bounded input/output edge cases:

- `load_image()` could enlarge an image smaller than 28 pixels before its boxes
  were scaled. It now preserves input dimensions; the paired image/box resize
  performs the transformation. Two regressions cover PIL/file/base64/URL inputs,
  tiny-image box alignment, unchanged regular-image mapping and caller-box preservation.
- `parse_answer()` could treat a region tag mentioned in prose as a positive
  selection. Bare-tag parsing now accepts only complete tag/separator lists;
  structured grounded output remains supported. Seven cases cover valid bare
  formats and negative, uncertain or quoted prose.

After both changes, **56 tests passed in 9.30 seconds**, with one upstream timm
warning. The no-download CPU preflight, dependency consistency and shell syntax
checks passed. See `docs/evidence/review-tests.txt` and
`docs/evidence/review-preflight.json`. Full pretrained inference and Docker
builds were not repeated in this review. The earlier 500 x 399 input and the
saved responses without region tags do not exercise the changed edge cases;
those measurements retain their original date and settings.

The first-person report was completed from the actual source and evidence,
and `docs/implementation_reference.md` records detailed technical facts for
independent review or subsequent report drafting. No personal identity details
or additional successful model runs were inferred.

## Evidence and reproduction

The submission includes `docs/evidence/cpu-preflight.json`,
`docs/evidence/cpu-acceptance.json` and `docs/evidence/cpu-acceptance.jpg`,
plus the test summary and stage timings. The separate `native-acceptance.*`
files and `native-environment.json` record the native check. Despite its
acceptance filename, the Docker result has a one-token cap, as recorded in
the JSON.
The acceptance JSON retains the raw model response and its limitations.
Its `outputs/...` paths refer to the original run; the copies are now in this
evidence directory.

For a fresh receiving computer, use the normal download-enabled acceptance
script rather than the staged-cache invocation above:

```bash
bash scripts/verify_container.sh cpu
```

It builds the image, checks imports, loads the checkpoints, runs inference
with a 16-token limit, and validates the result JSON and readable image. Downloads require internet
and approximately 11 GB of model files. See the README for RAM/disk planning.

## Limits

- No NVIDIA GPU was available. GPU image compilation, CUDA execution, UPN
  inference and 4-bit/8-bit quantization were not executed. The GPU override
  was configuration-checked only.
- The actual container run used Linux arm64. Linux x86_64 execution was not
  separately measured.
- The optional original decoder was exercised with small real architecture
  tests, not a complete pretrained baseline benchmark.
- Automatic approval review blocked an optional browser click test because
  the browser tool's usage limit was reached. Page inspection and automated
  input-validation tests completed; no full browser-driven inference is claimed.
- No fine-tuning or alignment training was performed. The bridge and region
  token meanings remain untrained. No benchmark score or validated object
  selection is claimed.


## Follow-up laptop execution: 17 September 2026, 32-token limit

At the user's request, the current source ran directly on the 8 GB Apple
Silicon Mac, using the existing Python environment and cached model weights.
The initial disk check showed about 28 GiB available. Disk availability fell
to about 18 GiB while running and recovered to about 24 GiB afterward.
Observed swap usage reached about 12 GiB. Other applications, including a
Claude virtual machine, were running, so these figures are whole-system
observations, not measurements of the model's memory requirement.

The no-download CPU preflight passed. The full pretrained test used the
bundled fruit image, its seven supplied boxes, seed 0, CPU execution, two
threads, offline model loading and a 32-token cap:

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 \
MPLCONFIGDIR=/private/tmp/fo1-mpl GRADIO_ANALYTICS_ENABLED=False \
  .venv/bin/python -B inference.py --device cpu --max-new-tokens 32 \
  --output outputs/laptop-check-2026-09-17/result.jpg \
  --json outputs/laptop-check-2026-09-17/result.json
```

The application completed its output-writing path. Its recorded request time
was **231.785 seconds**, including loading; inference took **194.022 seconds**.
Image processing took 35.5 seconds, region processing 34.2 seconds and generation
123.2 seconds. The saved JSON matched standard output, and the saved JPEG was
verified as a readable 500 x 399 image.

The complete decoded text at the cap was:

```text
Based on the visual analysis of the image provided:

1.  **Identify the object:** The image displays a **television** (or a similar
```

**This is a failed qualitative detection check.** The input shows fruit, not
a television, and the parsed detection list is empty. The answer is also
truncated by the token cap. The successful software execution must not be
reported as successful orange detection. This failure is consistent with the
known untrained visual-to-language bridge and region tokens; this single
example does not isolate the cause experimentally. A faster machine or more
disk space does not train those components.

The optional outer `/usr/bin/time -l` instrumentation returned status 1 because
the execution sandbox blocks `sysctl kern.clockrate`, after the application
had saved both outputs. A separate successful trivial Python command reproduced
the same timing-tool error. This timing-wrapper status is recorded separately
from the completed application output; the application's internal timings
above are used. The model was not rerun merely to remove the instrumentation.

Evidence: [raw JSON](evidence/laptop-32-token.json),
[output image](evidence/laptop-32-token.jpg), and
[stage timings](evidence/laptop-32-token-stages.txt). This was native macOS
execution of the reviewed source, not a new Docker or NVIDIA test.
