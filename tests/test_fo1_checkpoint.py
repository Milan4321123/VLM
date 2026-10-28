"""Regression checks for loading the vendored FO1 architecture under Transformers 5."""
from pathlib import Path
import json
import shutil

import torch

from vlm_fo1.model import OmChatQwen25VLConfig, OmChatQwen25VLForCausalLM

FIXTURE = Path(__file__).parent / 'fixtures' / 'fo1'


def test_full_architecture_matches_published_checkpoint_without_allocating_weights():
    config = OmChatQwen25VLConfig.from_pretrained(FIXTURE)
    config.name_or_path = str(FIXTURE)
    expected = set(json.loads((FIXTURE / 'model.safetensors.index.json').read_text())['weight_map'])
    with torch.device('meta'):
        full = OmChatQwen25VLForCausalLM(config)
        visual = OmChatQwen25VLForCausalLM(config, vision_only=True)
    assert set(full.state_dict()) == expected
    retained = set(visual.state_dict())
    assert retained < expected
    assert all(key.startswith(('model.layers.', 'model.embed_tokens.', 'model.norm.', 'lm_head.'))
               for key in expected - retained)
    assert sum(p.numel() for p in visual.parameters()) == 1064196928


def test_small_checkpoint_roundtrip_and_real_vision_forward(tmp_path, monkeypatch):
    shutil.copy(FIXTURE / 'preprocessor_config.json', tmp_path)
    config = OmChatQwen25VLConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        rope_scaling={'type': 'mrope', 'mrope_section': [3, 3, 2]},
        pad_token_id=0, eos_token_id=2, image_token_id=120, video_token_id=121,
        vision_start_token_id=122, mm_vision_tower='Qwen2.5-VL-tiny-test',
        mm_projector_type='identity', mm_hidden_size=32,
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=32, fullatt_block_indexes=[0]),
    )
    config.name_or_path = str(tmp_path)
    original = OmChatQwen25VLForCausalLM(config).eval()
    original.save_pretrained(tmp_path)

    materialize = OmChatQwen25VLForCausalLM._move_missing_keys_from_meta_to_device

    def poison_materialized_buffers(model, *args, **kwargs):
        materialize(model, *args, **kwargs)
        # Real empty_like allocations contain arbitrary data. Poison them to
        # make missing rotary initialization fail every time, not sporadically.
        for _, buffer in model.named_non_persistent_buffers():
            if buffer.is_floating_point():
                buffer.fill_(float('nan'))

    monkeypatch.setattr(OmChatQwen25VLForCausalLM, '_move_missing_keys_from_meta_to_device', poison_materialized_buffers)
    restored, info = OmChatQwen25VLForCausalLM.from_pretrained(
        tmp_path, vision_only=True, output_loading_info=True, attn_implementation='sdpa',
    )
    assert not info['missing_keys'] and not info['mismatched_keys']
    assert len(restored.model.layers) == 0
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[key])
    expected_buffers = dict(original.named_buffers())
    for key, buffer in restored.named_non_persistent_buffers():
        assert not buffer.is_meta and torch.isfinite(buffer).all(), key
        torch.testing.assert_close(buffer, expected_buffers[key])
    # Real vision patch embedding, rotary positions and attention after meta loading.
    pixels = [torch.zeros(16, 3 * 2 * 14 * 14)]
    grid = [torch.tensor([[1, 4, 4]])]
    with torch.inference_mode():
        original_features = original.encode_images(pixels, grid)[0][0]
        restored_features = restored.encode_images(pixels, grid)[0][0]
    assert restored_features.shape == (4, 32)
    assert torch.isfinite(restored_features).all()
    torch.testing.assert_close(restored_features, original_features)
