"""Command-line inference for VLM-FO1 with a free-form prompt.

Object proposals come from UPN unless you pass your own boxes with --boxes.

Examples (run from the repository root, or inside the Docker image):

    python cli.py --image demo/demo_image.jpg \
        --prompt "Please detect orange in this image. Answer the question with object indexes."

    python cli.py --image demo/demo_image_01.jpg \
        --task count --target "airplane with only one propeller"

    python cli.py --image demo/demo_image.jpg --task od --target "orange" \
        --boxes '[[161,11,292,127],[268,61,428,226]]'

The raw model answer and the boxes it selected are printed as JSON; the annotated
image is written to --output (default: outputs/<image name>_result.jpg).
"""
import argparse
import json
import os
import re
import sys

DEFAULT_MODEL = "omlab/VLM-FO1_Qwen2.5-VL-3B-v01"
LOCAL_MODEL_DIR = "./resources/VLM-FO1_Qwen2.5-VL-3B-v01"
DEFAULT_UPN_CKPT = "./resources/upn_large.pth"

# task name -> (template attribute in vlm_fo1.task_templates, needs --target)
TASKS = {
    "od": ("OD_template", True),
    "count": ("OD_Counting_template", True),
    "ocr": ("Region_OCR_template", True),
    "brief-caption": ("Brief_Region_Caption_template", True),
    "detailed-caption": ("Detailed_Region_Caption_template", True),
    "reason": ("Viusal_Region_Reasoning_template", True),
    "grounding": ("Grounding_template", False),
    "all": ("OD_All_template", False),
}


def default_model_path():
    env = os.environ.get("VLM_FO1_MODEL")
    if env:
        return env
    return LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else DEFAULT_MODEL


def build_parser():
    p = argparse.ArgumentParser(
        description="Run VLM-FO1 on one image with your own prompt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--image", required=True, help="Path or URL of the input image.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="Exact prompt sent to the model (used verbatim).")
    g.add_argument("--task", choices=sorted(TASKS), help="Build the prompt from a task template.")
    p.add_argument("--target", help="Text inserted into the task template (objects to find, question, ...).")
    p.add_argument("--boxes", help="JSON list of [x1, y1, x2, y2] boxes to use instead of UPN proposals.")
    p.add_argument("--region-tokens", action="store_true",
                   help="Append one <regionN> token per box to the prompt (what the Gradio demo does for hand-drawn regions).")
    p.add_argument("--threshold", type=float, default=0.3, help="Minimum UPN proposal score (default 0.3).")
    p.add_argument("--max-proposals", type=int, default=100, help="Keep at most this many UPN proposals (default 100).")
    p.add_argument("--model", default=default_model_path(),
                   help="VLM-FO1 checkpoint: Hugging Face id or local path (env VLM_FO1_MODEL).")
    p.add_argument("--upn-ckpt", default=os.environ.get("UPN_CKPT", DEFAULT_UPN_CKPT),
                   help="UPN checkpoint path (env UPN_CKPT).")
    p.add_argument("--device", default="cuda", help="Device for VLM-FO1 (default cuda).")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--output", help="Where to save the annotated image (default outputs/<name>_result.jpg).")
    p.add_argument("--json", dest="json_out", help="Also write the JSON result to this file.")
    p.add_argument("--no-image", action="store_true", help="Do not write the annotated image.")
    return p


def make_prompt(args):
    if args.prompt is not None:
        return args.prompt
    from vlm_fo1 import task_templates
    attr, needs_target = TASKS[args.task]
    template = getattr(task_templates, attr)
    if needs_target and not args.target:
        sys.exit(f"error: --task {args.task} needs --target")
    return template.format(args.target or "")


def parse_boxes(text):
    boxes = json.loads(text)
    if not isinstance(boxes, list) or any(len(b) != 4 for b in boxes):
        sys.exit("error: --boxes must be a JSON list of [x1, y1, x2, y2] lists")
    return [[float(v) for v in b] for b in boxes]


def get_proposals(image, args):
    from detect_tools.upn import UPNWrapper
    if not os.path.isfile(args.upn_ckpt):
        sys.exit(f"error: UPN checkpoint not found at {args.upn_ckpt} "
                 "(download it from https://github.com/IDEA-Research/ChatRex/releases/download/upn-large/upn_large.pth "
                 "or pass --boxes)")
    upn = UPNWrapper(args.upn_ckpt)
    proposals = upn.filter(upn.inference(image), min_score=args.threshold)
    return [[float(v) for v in box] for box in proposals["original_xyxy_boxes"][0][: args.max_proposals]]


def parse_answer(outputs, bboxes):
    """Map the model answer back to boxes. Returns (label -> indexes, detections list)."""
    from vlm_fo1.mm_utils import extract_predictions_to_indexes
    if "<ground>" in outputs:
        label_to_indexes = extract_predictions_to_indexes(outputs)
    else:
        found = re.findall(r"<region(\d+)>", outputs)
        label_to_indexes = {f"<region{m}>": {int(m)} for m in found}
    detections = []
    for label, indexes in label_to_indexes.items():
        for idx in sorted(indexes):
            if idx < len(bboxes):
                detections.append({"label": label, "region_index": idx, "bbox": bboxes[idx]})
    return detections


def main():
    args = build_parser().parse_args()
    prompt = make_prompt(args)

    import torch
    from vlm_fo1.mm_utils import draw_bboxes_and_save, load_image, prepare_inputs
    from vlm_fo1.model.builder import load_pretrained_model

    image = load_image(args.image)

    if args.boxes:
        bboxes = parse_boxes(args.boxes)
        source = "user"
    else:
        bboxes = get_proposals(image, args)
        source = "upn"
    if not bboxes:
        sys.exit("error: no object proposals; lower --threshold or pass --boxes")

    if args.region_tokens:
        prompt = prompt + "".join(f"<region{i}>" for i in range(len(bboxes)))

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "text", "text": prompt},
        ],
        "bbox_list": bboxes,
    }]

    tokenizer, model, image_processors = load_pretrained_model(args.model, device=args.device)
    generation_kwargs = prepare_inputs(
        args.model, model, image_processors, tokenizer, messages, device=args.device,
        max_tokens=args.max_tokens, top_p=0.05, temperature=0.0, do_sample=False,
    )
    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)
        outputs = tokenizer.decode(output_ids[0, generation_kwargs["inputs"].shape[1]:]).strip()

    detections = parse_answer(outputs, bboxes)
    result = {
        "image": args.image,
        "prompt": prompt,
        "proposal_source": source,
        "num_proposals": len(bboxes),
        "raw_output": outputs,
        "detections": detections,
    }

    if not args.no_image:
        output = args.output
        if not output:
            stem = os.path.splitext(os.path.basename(args.image.rstrip("/")))[0] or "image"
            output = os.path.join("outputs", f"{stem}_result.jpg")
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        label_to_boxes = {}
        for d in detections:
            label_to_boxes.setdefault(d["label"], []).append(d["bbox"])
        draw_bboxes_and_save(image=image.copy(), fo1_bboxes=label_to_boxes, output_path=output)
        result["output_image"] = output

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
