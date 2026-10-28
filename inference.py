"""Backwards-compatible fruit-image example, using the shared Qwen3.5 CLI."""
import argparse
import json
from pathlib import Path

from cli import main as cli_main
from vlm_fo1.runtime import DEFAULT_BBOXES, RuntimeConfig


def parse_args(argv=None):
    defaults = RuntimeConfig.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="demo/demo_image.jpg")
    parser.add_argument("--label", default="orange")
    parser.add_argument("--bboxes-json", help="JSON file containing manual boxes; bundled boxes fit only demo_image.jpg.")
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--llm-model-path", default=defaults.llm_model_path or "", help="Decoder checkpoint; pass '' for original decoder.")
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--load-4bit", action="store_true", default=defaults.load_4bit)
    parser.add_argument("--load-8bit", action="store_true", default=defaults.load_8bit)
    parser.add_argument("--max-new-tokens", type=int, default=defaults.max_tokens)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output", default="outputs/vlm_fo1_result.jpg")
    parser.add_argument("--json", dest="json_out")
    parser.add_argument("--no-image", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    translated = ["--image", args.image, "--task", "od", "--target", args.label,
                  "--model", args.model_path, "--llm-model-path", args.llm_model_path,
                  "--device", args.device, "--max-tokens", str(args.max_new_tokens),
                  "--seed", str(args.seed), "--output", args.output]
    if args.bboxes_json:
        translated.extend(["--boxes-file", args.bboxes_json])
    else:
        if Path(args.image).resolve() != (Path(__file__).parent / "demo/demo_image.jpg").resolve():
            raise SystemExit("error: provide --bboxes-json for an image other than the bundled fruit example.")
        translated.extend(["--boxes", json.dumps(DEFAULT_BBOXES)])
    if args.load_4bit:
        translated.append("--load-4bit")
    if args.load_8bit:
        translated.append("--load-8bit")
    if args.json_out:
        translated.extend(["--json", args.json_out])
    if args.no_image:
        translated.append("--no-image")
    return cli_main(translated)


if __name__ == "__main__":
    raise SystemExit(main())
