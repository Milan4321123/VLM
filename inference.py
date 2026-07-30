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


DEFAULT_BOXES = [
    [161.0, 11.0, 292.0, 127.0],
    [268.0, 61.0, 428.0, 226.0],
    [12.0, 100.0, 140.0, 227.0],
    [205.0, 188.0, 332.0, 320.0],
    [326.0, 202.0, 478.0, 357.0],
    [136.0, 106.0, 269.0, 233.0],
    [25.0, 206.0, 200.0, 383.0],
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="demo/demo_image.jpg")
    parser.add_argument("--label", default="orange")
    parser.add_argument("--boxes", help="JSON file containing a list of boxes")
    parser.add_argument(
        "--model-path",
        default="omlab/VLM-FO1_Qwen2.5-VL-3B-v01",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default="demo/vlm_fo1_result.jpg")
    return parser.parse_args()


def read_boxes(path):
    if path is None:
        return DEFAULT_BOXES

    with open(path, encoding="utf-8") as file:
        boxes = json.load(file)

    if not isinstance(boxes, list):
        raise ValueError("The boxes file must contain a JSON list")
    if any(not isinstance(box, list) or len(box) != 4 for box in boxes):
        raise ValueError("Each box must contain x1, y1, x2 and y2")
    return boxes


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Select a GPU or use --device cpu")

    boxes = read_boxes(args.boxes)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": args.image}},
                {"type": "text", "text": OD_template.format(args.label)},
            ],
            "bbox_list": boxes,
        }
    ]

    tokenizer, model, image_processors = load_pretrained_model(
        args.model_path,
        load_4bit=args.load_4bit,
        device=args.device,
    )
    generation_args = prepare_inputs(
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
        output_ids = model.generate(**generation_args)

    prompt_length = generation_args["inputs"].shape[1]
    output = tokenizer.decode(output_ids[0, prompt_length:]).strip()
    print("Model output:", output)

    selected_boxes = extract_predictions_to_bboxes(output, boxes)
    image = Image.open(args.image).convert("RGB")
    draw_bboxes_and_save(
        image=image,
        fo1_bboxes=selected_boxes,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
