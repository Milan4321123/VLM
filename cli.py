"""Run the experimental VLM-FO1 + Qwen3.5 decoder without fine-tuning.

CPU/Mac example (manual pixel-coordinate proposals):
    python cli.py --image demo/demo_image.jpg --task od --target orange \
        --boxes '[[161,11,292,127],[268,61,428,226]]' --device cpu

Dependency/device check, without downloading weights:
    python cli.py --check
"""
import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import sys
import time

from vlm_fo1.runtime import (
    DEFAULT_DECODER, DEFAULT_UPN_CKPT, RuntimeConfig,
    default_model_path, get_proposals as _get_proposals, get_runtime,
    load_input_image, parse_boxes, preflight, validate_boxes,
)

TASKS = {
    "od": ("OD_template", True), "count": ("OD_Counting_template", True),
    "ocr": ("Region_OCR_template", True), "brief-caption": ("Brief_Region_Caption_template", True),
    "detailed-caption": ("Detailed_Region_Caption_template", True),
    "reason": ("Viusal_Region_Reasoning_template", True),
    "grounding": ("Grounding_template", False), "all": ("OD_All_template", False),
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Check installed imports and device; no weights downloaded.")
    parser.add_argument("--image", help="Local image path or HTTP(S) URL.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prompt", help="Exact prompt sent to the model.")
    group.add_argument("--task", choices=sorted(TASKS), help="Build a prompt from a task template.")
    parser.add_argument("--target", help="Target objects or question for the task template.")
    boxes = parser.add_mutually_exclusive_group()
    boxes.add_argument("--boxes", help="JSON list of 1–100 [x1,y1,x2,y2] pixel-coordinate boxes; bypasses UPN.")
    boxes.add_argument("--boxes-file", help="JSON file containing manual boxes; bypasses UPN.")
    parser.add_argument("--region-tokens", action="store_true", help="Append one <regionN> per box to the prompt.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Minimum UPN proposal score (default 0.3).")
    parser.add_argument("--max-proposals", type=int, default=100, help="Maximum UPN proposals, 1–100.")
    parser.add_argument("--model", "--model-path", default=default_model_path(), help="Pretrained VLM-FO1 vision checkpoint.")
    decoder = parser.add_mutually_exclusive_group()
    decoder.add_argument("--llm-model-path", default=os.environ.get("VLM_FO1_LLM_MODEL", DEFAULT_DECODER),
                         help="Generation decoder (default Qwen/Qwen3.5-0.8B).")
    decoder.add_argument("--original-decoder", action="store_true", help="Use the original checkpoint decoder as a baseline.")
    parser.add_argument("--upn-ckpt", default=os.environ.get("UPN_CKPT", DEFAULT_UPN_CKPT))
    parser.add_argument("--device", default=os.environ.get("VLM_FO1_DEVICE", "auto"), help="auto (CUDA if available, otherwise CPU), cpu, cuda[:N], or mps.")
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument("--load-4bit", action="store_true", default=os.environ.get("VLM_FO1_LOAD_4BIT", "0") == "1", help="CUDA only; requires bitsandbytes.")
    quantization.add_argument("--load-8bit", action="store_true", default=os.environ.get("VLM_FO1_LOAD_8BIT", "0") == "1", help="CUDA only; requires bitsandbytes.")
    parser.add_argument("--max-tokens", "--max-new-tokens", type=int, default=int(os.environ.get("VLM_FO1_MAX_TOKENS", "256")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("VLM_FO1_SEED", "0")), help="Initialization and inference seed (default 0).")
    parser.add_argument("--output", help="Annotated image path (default outputs/<name>_result.jpg).")
    parser.add_argument("--json", dest="json_out", help="Also save the JSON result to this path.")
    parser.add_argument("--no-image", action="store_true", help="Do not save an annotated image.")
    return parser


def make_prompt(args):
    if args.prompt is not None:
        if not args.prompt.strip():
            raise ValueError("--prompt must not be empty.")
        return args.prompt
    if not args.task:
        raise ValueError("Provide --prompt or --task for inference.")
    from vlm_fo1 import task_templates
    attr, needs_target = TASKS[args.task]
    if needs_target and not (args.target and args.target.strip()):
        raise ValueError(f"--task {args.task} needs --target.")
    return getattr(task_templates, attr).format(args.target or "")


def get_proposals(image, args):
    return _get_proposals(image, args.threshold, args.max_proposals, args.upn_ckpt)


def _write_json(result, destination):
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = RuntimeConfig(
            model_path=args.model,
            llm_model_path=None if args.original_decoder else (args.llm_model_path or None),
            device=args.device, load_4bit=args.load_4bit, load_8bit=args.load_8bit,
            max_tokens=args.max_tokens, seed=args.seed,
        ).validate()
        if args.check:
            with redirect_stdout(sys.stderr):
                result = preflight(config)
            _write_json(result, args.json_out)
            return 0
        if not args.image:
            raise ValueError("--image is required for inference.")
        prompt = make_prompt(args)
        if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
            raise ValueError("--threshold must be between 0 and 1.")
        if not 1 <= args.max_proposals <= 100:
            raise ValueError("--max-proposals must be between 1 and 100.")
        # Reject malformed data before expensive imports, requests, or model loading.
        boxes = None
        if args.boxes is not None:
            boxes = parse_boxes(args.boxes)
        elif args.boxes_file:
            boxes = parse_boxes(Path(args.boxes_file).read_text(encoding="utf-8"))
        started = time.perf_counter()
        with redirect_stdout(sys.stderr):
            image = load_input_image(args.image)
            source = "user" if boxes is not None else "upn"
            boxes = validate_boxes(boxes, image.size) if boxes is not None else get_proposals(image, args)
            if args.region_tokens:
                prompt += "".join(f"<region{i}>" for i in range(len(boxes)))
            result = get_runtime(config).predict(image, boxes, prompt, source)
            result["image"] = args.image
            if not args.no_image:
                from vlm_fo1.mm_utils import draw_bboxes_and_save
                output = args.output or str(Path("outputs") / (Path(args.image).stem + "_result.jpg"))
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                labels = {}
                for detection in result["detections"]:
                    labels.setdefault(detection["label"], []).append(detection["bbox"])
                draw_bboxes_and_save(image=image.copy(), fo1_bboxes=labels, output_path=output)
                result["output_image"] = output
            result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        _write_json(result, args.json_out)
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
