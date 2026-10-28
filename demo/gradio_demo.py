"""Browser demo with lazy model loading and manual proposals for CPU/GPU use."""
from contextlib import redirect_stdout
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

# Also support `python demo/gradio_demo.py` from the repository checkout.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
from PIL import Image, ImageDraw

from vlm_fo1.runtime import (
    DEFAULT_BBOXES, DEFAULT_UPN_CKPT, EXPERIMENT_LIMITATION, RuntimeConfig,
    get_proposals, get_runtime, parse_boxes, validate_boxes,
)
from vlm_fo1.task_templates import (
    OD_template, OD_Counting_template, Viusal_Region_Reasoning_template,
    OD_All_template, Grounding_template,
)

TASK_TYPES = {
    "OD/REC": OD_template,
    "ODCounting": OD_Counting_template,
    "Region_OCR": "Please provide the ocr results of these regions in the image.",
    "Brief_Region_Caption": "Provide a brief description for these regions in the image.",
    "Detailed_Region_Caption": "Provide a detailed description for these regions in the image.",
    "Viusal_Region_Reasoning": Viusal_Region_Reasoning_template,
    "OD_All": OD_All_template,
    "Grounding": Grounding_template,
}
EXAMPLES = [
    ["demo_image.jpg", OD_template.format("orange, apple"), "OD/REC"],
    ["demo_image_01.jpg", OD_Counting_template.format("airplane with only one propeller"), "ODCounting"],
    ["demo_image_02.jpg", OD_template.format("the ball closest to the bear"), "OD/REC"],
    ["demo_image_03.jpg", OD_All_template, "OD_All"],
]
PROPOSAL_METHODS = ["Manual boxes", "Drawn regions", "Automatic UPN (CUDA)"]


def get_valid_examples():
    return [[str(Path(__file__).parent / name), prompt, task] for name, prompt, task in EXAMPLES
            if (Path(__file__).parent / name).is_file()]


def draw_bboxes(image, bboxes, labels=None):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for index, bbox in enumerate(bboxes):
        draw.rectangle(bbox, outline="red", width=3)
        draw.text((bbox[0] + 3, bbox[1] + 3), (labels or [])[index] if labels else f"region{index}", fill="red")
    return result


def extract_bbox_and_original_image(edited_image):
    """Each connected brush mark is a proposal; the untouched background is the model input."""
    if edited_image is None:
        return None, []
    if isinstance(edited_image, Image.Image):
        return edited_image, []
    if not isinstance(edited_image, dict):
        raise ValueError("Upload an image using the editor.")
    original = edited_image.get("background")
    if original is None:
        return None, []
    layers = edited_image.get("layers") or []
    if not layers:
        return original, []
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_dilation, disk
    mask = np.zeros((original.height, original.width), dtype=bool)
    for layer in layers:
        if layer.mode == "RGBA":
            mask |= np.asarray(layer.getchannel("A")) > 0
    regions = regionprops(label(binary_dilation(mask, disk(5))))
    return original, [[int(prop.bbox[1]), int(prop.bbox[0]), int(prop.bbox[3]), int(prop.bbox[2])]
                      for prop in regions]


def detect_model(image, threshold=0.3):
    return get_proposals(image, threshold, checkpoint=os.environ.get("UPN_CKPT", DEFAULT_UPN_CKPT))


def upload_defaults(value):
    """Gradio also fires upload when it initializes the example image."""
    image = value.get("background") if isinstance(value, dict) else value
    if image is not None:
        with Image.open(Path(__file__).parent / "demo_image.jpg") as fruit:
            rgb = image.convert("RGB")
            if rgb.size == fruit.size and rgb.tobytes() == fruit.convert("RGB").tobytes():
                return None, json.dumps(DEFAULT_BBOXES), "Manual boxes"
    return None, "", "Drawn regions"


def process(image, example_image, prompt, threshold, manual_boxes="", proposal_method="Manual boxes"):
    started = time.perf_counter()
    try:
        if not prompt or not prompt.strip():
            raise ValueError("Enter a prompt before submitting.")
        # Validate pasted JSON before invoking any ML code.
        boxes = parse_boxes(manual_boxes) if proposal_method == "Manual boxes" else None
        if proposal_method == "Drawn regions":
            image, drawn = extract_bbox_and_original_image(image)
            boxes = validate_boxes(drawn)
        elif isinstance(image, dict):
            image = image.get("background")
        if image is None:
            image = example_image
        if image is None:
            raise ValueError("Upload an image or load an example.")
        image = image.convert("RGB")
        source = "upn" if proposal_method == "Automatic UPN (CUDA)" else "user"
        with redirect_stdout(sys.stderr):
            boxes = detect_model(image, threshold) if source == "upn" else validate_boxes(boxes, image.size)
            if proposal_method == "Drawn regions":
                prompt += "".join(f"<region{i}>" for i in range(len(boxes)))
            result = get_runtime(RuntimeConfig.from_env()).predict(image, boxes, prompt, source)
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        result["proposal_method"] = proposal_method
        annotations = [(tuple(int(v) for v in item["bbox"]), item["label"]) for item in result["detections"]]
        return (image, annotations), draw_bboxes(image, boxes), result["raw_output"], result
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def create_demo():
    """Construct the UI without model imports, downloads, or GPU allocation."""
    config = RuntimeConfig.from_env().validate()
    examples = get_valid_examples()
    initial = examples[0][0] if examples else None
    with gr.Blocks(title="VLM-FO1 · Qwen3.5 experiment") as demo:
        gr.Markdown("# VLM-FO1 · Qwen3.5")
        gr.Markdown(f"**No-fine-tuning experiment.** {EXPERIMENT_LIMITATION}\n\n"
                    "Upload an image, paste boxes in pixel coordinates or draw separate regions, "
                    "and enter a prompt. Manual boxes and drawn regions work on CPU. "
                    "Automatic UPN proposals require NVIDIA CUDA. The bundled fruit example includes manual boxes.\n\n"
                    "The first submission downloads and loads the pretrained checkpoints; this can take several minutes. "
                    "Later submissions reuse the loaded model. A machine with sufficient RAM/disk is required.")
        gr.Markdown(f"**Configured decoder:** `{config.llm_model_path or config.model_path}` · "
                    f"**Device:** `{config.device}` · **Token limit:** `{config.max_tokens}` · **Seed:** `{config.seed}`")
        with gr.Row():
            with gr.Column():
                image_editor = gr.ImageEditor(
                    value=initial, label="Input image / draw separate regions", image_mode="RGBA", type="pil",
                    sources=["upload"], brush=gr.Brush(colors=["#FF0000"], color_mode="fixed", default_size=3),
                )
                proposal_method = gr.Radio(PROPOSAL_METHODS, value="Manual boxes", label="Proposal source")
                boxes_input = gr.Textbox(
                    value=json.dumps(DEFAULT_BBOXES) if initial else "", lines=4,
                    label="Manual boxes: [[x1,y1,x2,y2], ...]", info="1–100 boxes within the image. These defaults fit the fruit example only.",
                )
                template = gr.Dropdown(list(TASK_TYPES), value="OD/REC", label="Prompt template")
                prompt = gr.Textbox(value=OD_template.format("orange"), lines=3, label="Prompt")
                template.change(lambda selected: TASK_TYPES[selected].format("[WRITE YOUR INPUT HERE]"), template, prompt)
                threshold = gr.Slider(0, 1, value=0.3, step=0.01, label="UPN score threshold (automatic proposals only)")
                submit = gr.Button("Run inference", variant="primary")
                hidden_image = gr.Image(type="pil", image_mode="RGBA", visible=False)
                if examples:
                    gallery = gr.Examples(examples=examples, inputs=[hidden_image, prompt, template], label="Load an example")
                    def load_example(example, text, task):
                        fruit = task == "OD/REC" and text == OD_template.format("orange, apple")
                        return (example, json.dumps(DEFAULT_BBOXES) if fruit else "", "Manual boxes" if fruit else "Drawn regions")
                    gallery.load_input_event.then(load_example, [hidden_image, prompt, template], [image_editor, boxes_input, proposal_method])
                image_editor.upload(upload_defaults, inputs=[image_editor], outputs=[hidden_image, boxes_input, proposal_method])
            with gr.Column():
                proposals_image = gr.Image(label="Input proposals (red labels are region indexes)", height=300)
                result_image = gr.AnnotatedImage(label="Model-selected regions (experimental)", height=400)
                raw_output = gr.Textbox(label="Raw model output", lines=6)
                result_json = gr.JSON(label="Result and run metadata")
        submit.click(process, [image_editor, hidden_image, prompt, threshold, boxes_input, proposal_method],
                     [result_image, proposals_image, raw_output, result_json], concurrency_limit=1)
    return demo


# Existing callers used launch_demo() to construct the Blocks object.
launch_demo = create_demo


def main():
    demo = create_demo()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "8000"))),
        share=False, debug=False,
    )


if __name__ == "__main__":
    main()
