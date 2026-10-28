"""Fast offline checks for entry-point contracts and input validation."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import cli
from vlm_fo1.runtime import DEFAULT_DECODER, RuntimeConfig, parse_answer, parse_boxes, validate_boxes


@pytest.mark.parametrize("value", [
    "null", "{}", "[]", "[1]", "[[1,2,3]]", "[[1,2,3,4,5]]",
    '[["1",2,3,4]]', "[[true,2,3,4]]", "[[0,0,NaN,1]]", "[[0,0,Infinity,1]]",
    "[[0,0,0,1]]", "[[2,0,1,1]]", "[[-1,0,1,1]]", "[[0,-1,1,1]]",
    json.dumps([[0, 0, 1, 1]] * 101), "invalid json",
])
def test_bad_boxes_rejected(value):
    with pytest.raises(ValueError):
        parse_boxes(value)


def test_valid_boxes_are_floats_and_bounds_checked():
    assert parse_boxes("[[0,1,2,3]]") == [[0.0, 1.0, 2.0, 3.0]]
    assert len(validate_boxes([[0, 0, 10, 10]] * 100, (10, 10))) == 100
    with pytest.raises(ValueError, match="outside"):
        validate_boxes([[0, 0, 11, 10]], (10, 10))


def test_qwen35_default_and_original_decoder_option(monkeypatch):
    monkeypatch.delenv("VLM_FO1_LLM_MODEL", raising=False)
    monkeypatch.delenv("VLM_FO1_DEVICE", raising=False)
    args = cli.build_parser().parse_args([])
    assert args.llm_model_path == DEFAULT_DECODER
    assert args.device == "auto"
    assert cli.build_parser().parse_args(["--original-decoder"]).original_decoder


def test_malformed_boxes_fail_before_image_or_model_loading(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_input_image", lambda _: pytest.fail("image should not load"))
    monkeypatch.setattr(cli, "get_runtime", lambda _: pytest.fail("model should not load"))
    assert cli.main(["--image", "missing.jpg", "--task", "od", "--target", "cat", "--boxes", "[1]"]) == 1
    assert capsys.readouterr().out == ""


def test_prompt_target_validation():
    args = cli.build_parser().parse_args(["--task", "od"])
    with pytest.raises(ValueError, match="needs --target"):
        cli.make_prompt(args)
    assert "orange" in cli.make_prompt(cli.build_parser().parse_args(["--task", "od", "--target", "orange"]))


def test_answer_parser_ignores_bad_indexes_and_deduplicates():
    boxes = [[0, 0, 5, 5], [5, 5, 10, 10]]
    answer = "<ground>cat</ground><objects><region0><region0><region999></objects>"
    assert parse_answer(answer, boxes) == [{"label": "cat", "region_index": 0, "bbox": boxes[0]}]
    assert parse_answer("no regions", boxes) == []
    assert parse_answer("<region1> <region1> <region99>", boxes) == [
        {"label": "<region1>", "region_index": 1, "bbox": boxes[1]},
    ]


@pytest.mark.parametrize("answer", [
    "<region0><region1>", "[<region0>, <region1>]", "\n<region0>,\n<region1>\n",
])
def test_answer_parser_accepts_complete_bare_region_selections(answer):
    boxes = [[0, 0, 5, 5], [5, 5, 10, 10]]
    assert [item["region_index"] for item in parse_answer(answer, boxes)] == [0, 1]


@pytest.mark.parametrize("answer", [
    "Do not select <region0>.", "<region0> is not the target.",
    "The prompt mentions <region0>.", "Maybe <region0> or <region1>.",
])
def test_answer_parser_does_not_treat_prose_mentions_as_selections(answer):
    assert parse_answer(answer, [[0, 0, 5, 5], [5, 5, 10, 10]]) == []


def test_manual_cli_skips_upn_and_stdout_is_json(monkeypatch, capsys, tmp_path):
    from PIL import Image
    image = Image.new("RGB", (10, 10))
    observed = {}
    class FakeRuntime:
        def predict(self, image, boxes, prompt, source):
            print("library chatter")
            observed.update(boxes=boxes, source=source, prompt=prompt)
            return {"raw_output": "<region0>", "detections": parse_answer("<region0>", boxes)}
    def load(config):
        observed["config"] = config
        return FakeRuntime()
    monkeypatch.setattr(cli, "load_input_image", lambda _: image)
    monkeypatch.setattr(cli, "get_runtime", load)
    monkeypatch.setattr(cli, "get_proposals", lambda *_: pytest.fail("UPN must not run"))
    destination = tmp_path / "result.json"
    assert cli.main(["--image", "example.jpg", "--prompt", "find cat", "--boxes", "[[0,0,5,5]]",
                     "--no-image", "--json", str(destination), "--region-tokens"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert "library chatter" in captured.err
    assert result == json.loads(destination.read_text())
    assert result["elapsed_seconds"] >= 0
    assert observed["source"] == "user"
    assert observed["config"].llm_model_path == DEFAULT_DECODER
    assert observed["prompt"].endswith("<region0>")


def test_preflight_does_not_need_an_image(monkeypatch, capsys):
    monkeypatch.setattr(cli, "preflight", lambda config: {"status": "ok", "weights_loaded": False})
    monkeypatch.setattr(cli, "get_runtime", lambda _: pytest.fail("model should not load"))
    assert cli.main(["--check"]) == 0
    assert json.loads(capsys.readouterr().out)["weights_loaded"] is False


def test_help_does_not_import_torch_or_upn():
    code = "import cli,sys; cli.build_parser().format_help(); assert 'torch' not in sys.modules; assert 'detect_tools.upn' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])


def test_demo_creation_does_not_load_model(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    demo_module = importlib.import_module("demo.gradio_demo")
    monkeypatch.setattr(demo_module, "get_runtime", lambda _: pytest.fail("model should not load"))
    monkeypatch.setattr(demo_module, "get_proposals", lambda *_: pytest.fail("UPN should not load"))
    demo = demo_module.create_demo()
    assert demo is not None
    assert "detect_tools.upn" not in sys.modules
    demo.close()


def test_demo_manual_boxes_rejected_before_loading(monkeypatch):
    from demo import gradio_demo
    monkeypatch.setattr(gradio_demo, "get_runtime", lambda _: pytest.fail("model should not load"))
    with pytest.raises(gradio_demo.gr.Error):
        gradio_demo.process(None, None, "find cat", 0.3, "[1]", "Manual boxes")


def test_cpu_upn_has_actionable_error_without_import(monkeypatch):
    import torch
    from vlm_fo1.runtime import _load_upn
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="manual bounding boxes"):
        _load_upn("nonexistent-upn.pth")
    assert "detect_tools.upn" not in sys.modules


def test_demo_preserves_fruit_boxes_on_initial_upload_and_clears_other_images():
    from PIL import Image
    from demo.gradio_demo import upload_defaults
    with Image.open(Path(__file__).resolve().parents[1] / 'demo/demo_image.jpg') as fruit:
        _, boxes, method = upload_defaults({'background': fruit.convert('RGBA')})
    assert len(json.loads(boxes)) == 7 and method == 'Manual boxes'
    assert upload_defaults(Image.new('RGB', (10, 10))) == (None, '', 'Drawn regions')


def test_requested_image_write_failure_is_not_reported_as_success(monkeypatch, capsys, tmp_path):
    from PIL import Image
    image = Image.new('RGB', (10, 10))
    class Runtime:
        def predict(self, *args):
            return {'raw_output': 'test', 'detections': []}
    monkeypatch.setattr(cli, 'load_input_image', lambda _: image)
    monkeypatch.setattr(cli, 'get_runtime', lambda _: Runtime())
    bad_output = tmp_path / 'directory.jpg'
    bad_output.mkdir()
    result_path = tmp_path / 'result.json'
    result = cli.main(['--image', 'test.jpg', '--prompt', 'describe', '--boxes', '[[0,0,5,5]]',
                       '--output', str(bad_output), '--json', str(result_path)])
    assert result == 1 and capsys.readouterr().out == ''
    assert not result_path.exists()
