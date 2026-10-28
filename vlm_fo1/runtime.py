"""Shared inference runtime. Importing this module never downloads or loads weights."""
from dataclasses import dataclass
from functools import lru_cache
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

DEFAULT_MODEL = "omlab/VLM-FO1_Qwen2.5-VL-3B-v01"
DEFAULT_DECODER = "Qwen/Qwen3.5-0.8B"
DEFAULT_UPN_CKPT = "./resources/upn_large.pth"
UPN_URL = "https://github.com/IDEA-Research/ChatRex/releases/download/upn-large/upn_large.pth"
EXPERIMENT_LIMITATION = (
    "Experimental Qwen3.5 decoder replacement without fine-tuning. The feature "
    "bridge and added region-token embeddings are untrained. Generated text and "
    "region selections are not validated object-detection or counting results."
)
DEFAULT_BBOXES = [
    [161, 11, 292, 127], [268, 61, 428, 226], [12, 100, 140, 227],
    [205, 188, 332, 320], [326, 202, 478, 357], [136, 106, 269, 233],
    [25, 206, 200, 383],
]


def default_model_path():
    if os.environ.get("VLM_FO1_MODEL"):
        return os.environ["VLM_FO1_MODEL"]
    for local in ("./resources/VLM-FO1-3B-v01", "./resources/VLM-FO1_Qwen2.5-VL-3B-v01"):
        if os.path.isfile(os.path.join(local, "config.json")):
            return local
    return DEFAULT_MODEL


@dataclass(frozen=True)
class RuntimeConfig:
    model_path: str = DEFAULT_MODEL
    llm_model_path: str | None = DEFAULT_DECODER
    device: str = "auto"
    load_4bit: bool = False
    load_8bit: bool = False
    max_tokens: int = 256
    seed: int = 0

    @classmethod
    def from_env(cls):
        return cls(
            model_path=default_model_path(),
            llm_model_path=os.environ.get("VLM_FO1_LLM_MODEL", DEFAULT_DECODER) or None,
            device=os.environ.get("VLM_FO1_DEVICE", "auto"),
            load_4bit=os.environ.get("VLM_FO1_LOAD_4BIT", "0") == "1",
            load_8bit=os.environ.get("VLM_FO1_LOAD_8BIT", "0") == "1",
            max_tokens=int(os.environ.get("VLM_FO1_MAX_TOKENS", "256")),
            seed=int(os.environ.get("VLM_FO1_SEED", "0")),
        )

    def validate(self):
        if not self.model_path.strip():
            raise ValueError("A vision checkpoint is required.")
        if self.llm_model_path is not None and not self.llm_model_path.strip():
            raise ValueError("A decoder checkpoint is required, or choose the original decoder.")
        if not re.fullmatch(r"auto|cpu|cuda(?::\d+)?|mps", self.device):
            raise ValueError("--device must be auto, cpu, cuda, cuda:N, or mps.")
        if self.load_4bit and self.load_8bit:
            raise ValueError("Choose either 4-bit or 8-bit loading.")
        if not 1 <= self.max_tokens <= 8192:
            raise ValueError("--max-tokens must be between 1 and 8192.")
        if not 0 <= self.seed < 2**32:
            raise ValueError("--seed must be between 0 and 4294967295.")
        return self


def validate_boxes(boxes, image_size=None):
    """Validate pixel-coordinate xyxy boxes without importing ML dependencies."""
    if not isinstance(boxes, list) or not 1 <= len(boxes) <= 100:
        raise ValueError("Bounding boxes must be a JSON list containing 1 to 100 boxes.")
    result = []
    for index, box in enumerate(boxes):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"Box {index} must contain [x1, y1, x2, y2].")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in box):
            raise ValueError(f"Box {index} coordinates must be numbers, not strings or booleans.")
        try:
            coords = [float(v) for v in box]
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"Box {index} coordinates must be finite numbers.") from exc
        if not all(math.isfinite(v) for v in coords):
            raise ValueError(f"Box {index} coordinates must be finite numbers.")
        x1, y1, x2, y2 = coords
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError(f"Box {index} requires 0 <= x1 < x2 and 0 <= y1 < y2.")
        if image_size and (x2 > image_size[0] or y2 > image_size[1]):
            raise ValueError(f"Box {index} extends outside the {image_size[0]}x{image_size[1]} image.")
        result.append(coords)
    return result


def parse_boxes(text):
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Bounding boxes must be valid JSON: [[x1, y1, x2, y2], ...].") from exc
    return validate_boxes(value)


def resolve_device(requested="auto"):
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Use --device cpu and provide manual --boxes.")
        if ":" in device and int(device.split(":")[1]) >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device {device} is unavailable.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Use --device cpu.")
    return device


def _validate_quantization(config, device):
    if config.load_4bit or config.load_8bit:
        if not device.startswith("cuda"):
            raise ValueError("4-bit and 8-bit loading require CUDA. Disable quantization on CPU/Mac.")
        if importlib.util.find_spec("bitsandbytes") is None:
            raise ValueError("Quantization requires bitsandbytes. Install requirements-gpu.txt.")


def preflight(config):
    """Exercise installed inference imports and device detection, without weights."""
    config.validate()
    versions = {}
    for name in ("torch", "torchvision", "transformers", "gradio"):
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "unknown")
    importlib.import_module("vlm_fo1.model.builder")
    importlib.import_module("vlm_fo1.model.language_model.omchat_qwen2_5_vl")
    importlib.import_module("vlm_fo1.mm_utils")
    device = resolve_device(config.device)
    _validate_quantization(config, device)
    return {
        "status": "ok", "check": "imports_and_device", "weights_loaded": False,
        "device": device, "versions": versions,
        "model": {"vision_checkpoint": config.model_path,
                  "decoder_checkpoint": config.llm_model_path or config.model_path},
        "bridge_is_trained": False if config.llm_model_path else None,
        "limitations": [EXPERIMENT_LIMITATION] if config.llm_model_path else [],
        "next_step": "Run inference with an image and manual boxes to verify pretrained weights.",
    }


def load_input_image(source):
    from PIL import Image, ImageOps
    from io import BytesIO
    from urllib.request import urlopen
    if isinstance(source, Image.Image):
        return ImageOps.exif_transpose(source).convert("RGB")
    if str(source).startswith(("https://", "http://")):
        with urlopen(source, timeout=30) as response:
            with Image.open(BytesIO(response.read())) as image:
                return ImageOps.exif_transpose(image).convert("RGB")
    with Image.open(source) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


@lru_cache(maxsize=1)
def _load_upn(checkpoint):
    # The vendored UPN has CUDA-only deformable attention operations.
    import torch
    if not torch.cuda.is_available():
        raise ValueError("Automatic UPN proposals require an NVIDIA CUDA GPU. On Mac/CPU, "
                         "draw regions or supply manual bounding boxes (--boxes).")
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        if os.environ.get("UPN_AUTO_DOWNLOAD", "0") != "1":
            raise ValueError(f"UPN checkpoint is missing: {checkpoint}. Provide manual boxes, "
                             "or set UPN_AUTO_DOWNLOAD=1 to download it when UPN is requested.")
        from urllib.request import urlopen
        import shutil
        import tempfile
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=checkpoint.parent, delete=False) as output:
                temporary = output.name
                print(f"Downloading UPN checkpoint to {checkpoint} ...", file=sys.stderr)
                with urlopen(UPN_URL, timeout=60) as response:
                    shutil.copyfileobj(response, output)
            os.replace(temporary, checkpoint)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
    from detect_tools.upn import UPNWrapper
    return UPNWrapper(str(checkpoint))


def get_proposals(image, threshold=0.3, max_proposals=100, checkpoint=DEFAULT_UPN_CKPT):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1.")
    if not 1 <= max_proposals <= 100:
        raise ValueError("--max-proposals must be between 1 and 100.")
    upn = _load_upn(str(checkpoint))
    proposals = upn.filter(upn.inference(image), min_score=threshold)
    boxes = [[float(v) for v in box] for box in proposals["original_xyxy_boxes"][0][:max_proposals]]
    if not boxes:
        raise ValueError("UPN returned no proposals. Lower the threshold or provide manual boxes.")
    # UPN can predict slightly outside the image. Bound its proposals to valid pixels.
    width, height = image.size
    clipped = [[max(0., min(width, b[0])), max(0., min(height, b[1])),
                max(0., min(width, b[2])), max(0., min(height, b[3]))] for b in boxes]
    clipped = [b for b in clipped if b[0] < b[2] and b[1] < b[3]]
    return validate_boxes(clipped, image.size)


def parse_answer(output, boxes):
    """Parse model region tags; unrecognized/out-of-range tags are never detections."""
    labels = {}
    grounded = re.findall(r"<ground>(.*?)</ground>\s*<objects>(.*?)</objects>", output, re.DOTALL)
    if grounded:
        for label, tags in grounded:
            labels.setdefault(label.strip(), set()).update(int(v) for v in re.findall(r"<region(\d+)>", tags))
    elif re.fullmatch(r"[\s,\[\]]*(?:<region\d+>[\s,\[\]]*)+", output.strip()):
        # A tag mentioned in prose may be negated or merely quoted. Only a
        # complete list of region tags counts as an unlabelled selection.
        for value in re.findall(r"<region(\d+)>", output):
            labels.setdefault(f"<region{value}>", set()).add(int(value))
    return [{"label": label, "region_index": index, "bbox": boxes[index]}
            for label, indexes in labels.items() for index in sorted(indexes) if 0 <= index < len(boxes)]


def seed_inference(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


class InferenceRuntime:
    def __init__(self, config):
        config.validate()
        self.config = config
        self.device = resolve_device(config.device)
        _validate_quantization(config, self.device)
        seed_inference(config.seed)
        from vlm_fo1.model.builder import load_pretrained_model
        self.tokenizer, self.model, self.image_processors = load_pretrained_model(
            config.model_path, device=self.device, llm_model_path=config.llm_model_path,
            load_4bit=config.load_4bit, load_8bit=config.load_8bit,
            bridge_seed=config.seed,
        )

    def predict(self, image, boxes, prompt, proposal_source="user"):
        start = time.perf_counter()
        boxes = validate_boxes(boxes, image.size)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Enter a non-empty prompt.")
        import torch
        from vlm_fo1.mm_utils import prepare_inputs
        seed_inference(self.config.seed)
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "text", "text": prompt},
        ], "bbox_list": boxes}]
        generation_kwargs = prepare_inputs(
            self.config.model_path, self.model, self.image_processors, self.tokenizer,
            messages, device=self.device, max_tokens=self.config.max_tokens,
            temperature=0.0, do_sample=False, stream=False, verbose=False,
        )
        with torch.inference_mode():
            output_ids = self.model.generate(**generation_kwargs)
            output = self.tokenizer.decode(
                output_ids[0, generation_kwargs["inputs"].shape[1]:],
            ).strip()
        return {
            "prompt": prompt, "proposal_source": proposal_source, "num_proposals": len(boxes),
            "proposals": boxes, "raw_output": output, "detections": parse_answer(output, boxes),
            "model": {"vision_checkpoint": self.config.model_path,
                      "decoder_checkpoint": self.config.llm_model_path or self.config.model_path},
            "device": self.device, "seed": self.config.seed, "do_sample": False,
            "max_new_tokens": self.config.max_tokens,
            "bridge_is_trained": False if self.config.llm_model_path else None,
            "inference_status": getattr(self.model, "inference_status", {
                "backend": "original_fo1", "fine_tuning_performed": False,
            }),
            "limitations": [EXPERIMENT_LIMITATION] if self.config.llm_model_path else [],
            "inference_seconds": round(time.perf_counter() - start, 3),
        }


@lru_cache(maxsize=1)
def get_runtime(config):
    """Cache the single demo model; initialization occurs only on submission."""
    return InferenceRuntime(config)
