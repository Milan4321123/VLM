import argparse
import json

import torch
from PIL import Image

from vlm_fo1.model.builder import load_pretrained_model
from vlm_fo1.mm_utils import (
    prepare_inputs,
    draw_bboxes_and_save,
    extract_predictions_to_bboxes,
)
from vlm_fo1.task_templates import OD_template


DEFAULT_BBOXES = [
    [161.0, 11.0, 292.0, 127.0],
    [268.0, 61.0, 428.0, 226.0],
    [12.0, 100.0, 140.0, 227.0],
    [205.0, 188.0, 332.0, 320.0],
    [326.0, 202.0, 478.0, 357.0],
    [136.0, 106.0, 269.0, 233.0],
    [25.0, 206.0, 200.0, 383.0],
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run VLM-FO1 inference with provided bounding boxes.")
    parser.add_argument("--image", default="demo/demo_image.jpg")
    parser.add_argument("--label", default="orange")
    parser.add_argument("--bboxes-json", help="Optional JSON file containing [[x1, y1, x2, y2], ...].")
    parser.add_argument("--model-path", default="omlab/VLM-FO1_Qwen2.5-VL-3B-v01")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-4bit", action="store_true", help="Reduce VRAM use; recommended for Colab T4.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default="demo/vlm_fo1_result.jpg")
    return parser.parse_args()


def load_bboxes(path):
    if path is None:
        return DEFAULT_BBOXES
    with open(path, "r", encoding="utf-8") as handle:
        bboxes = json.load(handle)
    if not isinstance(bboxes, list) or any(len(box) != 4 for box in bboxes):
        raise ValueError("Bounding box JSON must contain [[x1, y1, x2, y2], ...].")
    return bboxes


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is available. Enable a GPU runtime in Colab.")

    bbox_list = load_bboxes(args.bboxes_json)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": args.image}},
                {"type": "text", "text": OD_template.format(args.label)},
            ],
            "bbox_list": bbox_list,
        }
    ]

    tokenizer, model, image_processors = load_pretrained_model(
        args.model_path,
        load_4bit=args.load_4bit,
        device=args.device,
    )
    generation_kwargs = prepare_inputs(
        args.model_path,
        model,
        image_processors,
        tokenizer,
        messages,
        device=args.device,
        max_tokens=args.max_new_tokens,
        top_p=0.05,
        temperature=0.0,
        do_sample=False,
    )

    with torch.inference_mode():
        output_ids = model.generate(**generation_kwargs)
        outputs = tokenizer.decode(
            output_ids[0, generation_kwargs["inputs"].shape[1]:]
        ).strip()

    print("Model output:", outputs)
    bboxes = extract_predictions_to_bboxes(outputs, bbox_list)
    image = Image.open(args.image).convert("RGB")
    draw_bboxes_and_save(image=image, fo1_bboxes=bboxes, output_path=args.output)


if __name__ == "__main__":
    main()
