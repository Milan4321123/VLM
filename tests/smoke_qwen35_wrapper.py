"""
CPU smoke check for Qwen35FO1InferenceWrapper. Not a pytest module — run it
directly:  python tests/smoke_qwen35_wrapper.py

Loads the real Qwen/Qwen3.5-0.8B tokenizer and weights (about 1.7 GB on the
first run, cached afterwards) and drives the actual wrapper end to end. The
big FO1 vision stack is replaced by a stub that returns randomly initialised
but correctly shaped 2048-dim features, so this checks everything the swap
introduced — the region-token setup, the 2048->1024 bridge, the embedding
splice at the image/region sentinels, and a real generate() call against
Qwen3.5's architecture — without needing a GPU or the 6GB checkpoint.

The token setup below mirrors builder._ensure_qwen35_fo1_tokens and
builder._load_qwen35_language_model rather than importing them, because
importing the builder drags in the whole vision stack; keep them in sync.

The decoded output is expected to be fluent but ungrounded: the bridge is
untrained, so the region features reaching Qwen3.5 carry no real signal.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForImageTextToText

from vlm_fo1.model.language_model.qwen35_fo1_wrapper import Qwen35FO1InferenceWrapper
from vlm_fo1.constants import IMAGE_TOKEN_INDEX, DEFAULT_REGION_INDEX

VISION_HIDDEN = 2048
NUM_IMAGE_TOKENS = 4
NUM_REGION_TOKENS = 2


class FakeConfig:
    hidden_size = VISION_HIDDEN
    patch_size = 14


class FakeVisionTower:
    def __init__(self):
        self.config = FakeConfig()


class FakeVisionBackbone(nn.Module):
    """Stands in for the FO1 checkpoint after decoder stripping."""

    def __init__(self):
        super().__init__()
        self.config = FakeConfig()
        self._vt = FakeVisionTower()

    def get_vision_tower(self):
        return self._vt

    def get_vision_tower_aux(self):
        return self._vt  # truthy, so the wrapper also exercises region features

    def encode_images(self, images, image_grid_thws=None):
        image_features = [torch.randn(NUM_IMAGE_TOKENS, VISION_HIDDEN) for _ in images]
        grid_thws = [torch.tensor([[1, 4, 4]]) for _ in images]
        multi_level = [None for _ in images]
        return image_features, grid_thws, multi_level

    def encode_regions(self, images_aux, bbox_list, vt_multi_level_features, vt_images_size):
        return [torch.randn(NUM_REGION_TOKENS, VISION_HIDDEN) for _ in images_aux]


def build_prompt_ids(tokenizer):
    prefix = tokenizer.encode("<|im_start|>user\ndescribe ", add_special_tokens=False)
    suffix = tokenizer.encode(" what is in region0 and region1?<|im_end|>\n", add_special_tokens=False)
    ids = prefix + [IMAGE_TOKEN_INDEX] + [DEFAULT_REGION_INDEX] * NUM_REGION_TOKENS + suffix
    return torch.tensor([ids], dtype=torch.long)


def main():
    model_id = "Qwen/Qwen3.5-0.8B"
    print(f"Loading tokenizer from {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    added = tokenizer.add_tokens([f"<region{i}>" for i in range(100)], special_tokens=True)
    print(f"Added {added} region tokens to tokenizer, vocab size now {len(tokenizer)}")

    print(f"Loading {model_id} weights on CPU ...")
    language_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
    )
    if len(tokenizer) > language_model.get_input_embeddings().num_embeddings:
        language_model.resize_token_embeddings(len(tokenizer))
    assert len(tokenizer) <= language_model.get_input_embeddings().num_embeddings
    print("Language model loaded; embedding table covers the tokenizer.")

    wrapper = Qwen35FO1InferenceWrapper(
        vision_backbone=FakeVisionBackbone(),
        language_model=language_model,
        tokenizer=tokenizer,
    )
    wrapper.eval()
    print(f"Wrapper constructed. bridge_is_trained={wrapper.bridge_is_trained}, "
          f"vision_hidden={wrapper.vision_hidden_size}, language_hidden={wrapper.language_hidden_size}")

    input_ids = build_prompt_ids(tokenizer)
    print(f"Prompt token length (with sentinels): {input_ids.shape[1]}")

    images = [torch.randn(3, 448, 448)]
    images_aux = [torch.randn(3, 448, 448)]
    bbox_list = [torch.tensor([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 100.0]])]

    with torch.inference_mode():
        output_ids = wrapper.generate(
            inputs=input_ids,
            images=images,
            images_aux=images_aux,
            bbox_list=bbox_list,
            image_grid_thws=[torch.tensor([[1, 4, 4]])],
            max_new_tokens=16,
            do_sample=False,
        )

    print("generate() returned shape:", tuple(output_ids.shape))
    new_tokens = output_ids[0, input_ids.shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print("Decoded new tokens:", repr(decoded))
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        main()
