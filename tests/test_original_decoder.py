"""Exercise the original decoder's real attention and incremental HF5 cache."""
from pathlib import Path
import shutil

import torch

from vlm_fo1.constants import IMAGE_TOKEN_INDEX
from vlm_fo1.model import OmChatQwen25VLConfig, OmChatQwen25VLForCausalLM


def tiny_original_model(tmp_path):
    fixture = Path(__file__).parent / 'fixtures' / 'fo1'
    shutil.copy(fixture / 'preprocessor_config.json', tmp_path)
    config = OmChatQwen25VLConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        rope_scaling={'type': 'mrope', 'mrope_section': [3, 3, 2]},
        bos_token_id=1, pad_token_id=0, eos_token_id=2, image_token_id=120, video_token_id=121,
        vision_start_token_id=122, mm_vision_tower='Qwen2.5-VL-tiny-test',
        mm_projector_type='identity', mm_hidden_size=32,
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=32, fullatt_block_indexes=[0]),
    )
    config.name_or_path = str(tmp_path)
    return OmChatQwen25VLForCausalLM(config).eval()


def test_original_visual_generation_uses_dynamic_cache(tmp_path):
    torch.manual_seed(0)
    model = tiny_original_model(tmp_path)
    prompt = torch.tensor([[1, 122, IMAGE_TOKEN_INDEX, 4]])
    with torch.inference_mode():
        generated = model.generate(
            input_ids=prompt, attention_mask=torch.ones_like(prompt),
            images=[torch.zeros(16, 3 * 2 * 14 * 14)],
            image_grid_thws=[torch.tensor([[1, 4, 4]])],
            max_new_tokens=3, eos_token_id=[], do_sample=False, use_cache=True,
            return_dict_in_generate=True, output_scores=True,
        )
    assert generated.sequences.shape == (1, 7)
    assert torch.equal(generated.sequences[:, :4], prompt)
    assert len(generated.scores) == 3
    assert all(torch.isfinite(score).all() for score in generated.scores)
    # Four visual patches replace the marker (+3), then two continuation tokens
    # have been forwarded when the third and final token is selected.
    assert generated.past_key_values.get_seq_length() == 9


def test_original_text_cached_logits_match_full_sequence(tmp_path):
    model = tiny_original_model(tmp_path)
    prompt = torch.tensor([[1, 4, 5]])
    with torch.inference_mode():
        prefill = model(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                        use_cache=True, cache_position=torch.arange(3))
        continued = model(input_ids=torch.tensor([[6]]), attention_mask=torch.ones(1, 4, dtype=torch.long),
                          past_key_values=prefill.past_key_values, use_cache=True,
                          cache_position=torch.tensor([3]))
        uncached = model(input_ids=torch.tensor([[1, 4, 5, 6]]), attention_mask=torch.ones(1, 4, dtype=torch.long),
                         use_cache=False, cache_position=torch.arange(4))
    assert torch.isfinite(continued.logits).all()
    torch.testing.assert_close(continued.logits[:, -1], uncached.logits[:, -1], atol=1e-6, rtol=1e-5)
