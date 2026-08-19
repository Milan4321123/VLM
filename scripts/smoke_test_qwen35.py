"""
End-to-end wiring test for the Qwen3.5 backbone, with no downloads and no GPU.

Builds a randomly initialized FO1-on-Qwen3.5 model with miniature dimensions
and pushes one image plus two boxes through the whole pipeline: prompt
splicing (-200 / -300 placeholders), the Qwen3.5 vision tower with level
hooks, the DaViT aux tower, the HFRE region encoder, both projectors, mrope
position ids and a few steps of generate(). Output quality is meaningless
(random weights) - what this verifies is that every tensor has the right
shape and every module is wired correctly.

Run from the repo root:
  python scripts/smoke_test_qwen35.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from PIL import Image
import numpy as np

from vlm_fo1.model.language_model.omchat_qwen3_5 import OmChatQwen35Config, OmChatQwen35ForCausalLM
from vlm_fo1.constants import IMAGE_TOKEN_INDEX, DEFAULT_REGION_INDEX


def build_tiny_model():
    vision_config = dict(
        depth=4,
        hidden_size=32,
        out_hidden_size=64,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=256,
    )
    text_config = dict(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=0,
    )
    # region feature width in concat mode:
    #   DaViT pyramid channels (256+512+1024+2048) + 4 hooked ViT levels x 32
    mm_region_hidden_size = (256 + 512 + 1024 + 2048) + 4 * vision_config["hidden_size"]

    config = OmChatQwen35Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=999,
        video_token_id=998,
        mm_vision_tower="qwen3.5-tiny",
        mm_vision_tower_aux="davit-large",
        mm_projector_type="mlp2x_gelu",
        mm_projector_aux_type="mlp2x_gelu",
        mm_hidden_size=vision_config["out_hidden_size"],
        mm_region_hidden_size=mm_region_hidden_size,
        mm_use_vision_tower_region_feature=True,
        mm_use_region_index_token=True,
        mm_roi_output_size=7,
        mm_apply_position_embedding=True,
        mm_pos_embedding_strategy="bbox_based",
        mm_region_feature_combination="concat",
        mm_apply_region_layer_norm=True,
        aux_image_size=768,
        aux_image_aspect_ratio="squash",
        delay_load=True,
    )
    torch.manual_seed(0)
    model = OmChatQwen35ForCausalLM(config)
    model.eval()
    return model


def main():
    print("building tiny model (random weights)...")
    model = build_tiny_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model built: {n_params/1e6:.1f}M params")

    # one random image, preprocessed by both towers' processors
    pil = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))

    primary_processor = model.get_vision_tower().image_processor
    out = primary_processor(images=pil, return_tensors="pt")
    pixel_values, grid_thw = out["pixel_values"], out["image_grid_thw"]
    print(f"primary pixel_values {tuple(pixel_values.shape)}, grid_thw {grid_thw.tolist()}")

    aux_processor = model.get_vision_tower_aux().image_processor
    aux_pixel_values = aux_processor.preprocess(pil, return_tensors="pt")["pixel_values"][0]
    print(f"aux pixel_values {tuple(aux_pixel_values.shape)}")

    # two boxes in aux-image coordinates
    bbox_list = [torch.tensor([[10.0, 10.0, 200.0, 180.0], [300.0, 250.0, 600.0, 560.0]])]

    # hand-built prompt: text ids with an image slot (-200) and one region
    # feature slot (-300) per box, no tokenizer needed
    ids = [5, 6, 7, IMAGE_TOKEN_INDEX, 8,
           DEFAULT_REGION_INDEX, 9, DEFAULT_REGION_INDEX, 10, 11]
    input_ids = torch.tensor([ids], dtype=torch.long)

    print("running generate()...")
    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=[pixel_values],
            image_grid_thws=[grid_thw],
            images_aux=[aux_pixel_values],
            bbox_list=bbox_list,
            do_sample=False,
            max_new_tokens=4,
            use_cache=True,
        )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    print(f"generated token ids: {new_tokens.tolist()}")

    # also exercise the no-region path
    print("running generate() without boxes...")
    ids_noreg = [5, 6, 7, IMAGE_TOKEN_INDEX, 8, 9]
    with torch.inference_mode():
        output_ids = model.generate(
            inputs=torch.tensor([ids_noreg], dtype=torch.long),
            images=[pixel_values],
            image_grid_thws=[grid_thw],
            images_aux=None,
            bbox_list=None,
            do_sample=False,
            max_new_tokens=4,
            use_cache=True,
        )
    print(f"generated token ids: {output_ids[0, len(ids_noreg):].tolist()}")

    # round-trip through save_pretrained / from_pretrained - this is the same
    # mechanism a real checkpoint folder uses, so it catches config
    # serialization problems and wrong weight prefixes
    import tempfile
    print("checkpoint round-trip...")
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = os.path.join(tmp, "VLM-FO1_Qwen3.5-tiny")
        model.save_pretrained(ckpt_dir)
        reloaded, loading_info = OmChatQwen35ForCausalLM.from_pretrained(
            ckpt_dir, output_loading_info=True)
        missing = [k for k in loading_info["missing_keys"]]
        unexpected = [k for k in loading_info["unexpected_keys"]]
        assert not missing, f"missing keys after reload: {missing[:5]}..."
        assert not unexpected, f"unexpected keys after reload: {unexpected[:5]}..."
    print("checkpoint round-trip ok (no missing or unexpected keys)")

    print("smoke test passed: splice, towers, HFRE, projectors and mrope all wired")


if __name__ == "__main__":
    main()
