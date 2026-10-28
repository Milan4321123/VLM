"""The official text loader must preserve Qwen generation and exclude native vision."""
import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

from vlm_fo1.model.builder import _load_qwen35_language_model


@pytest.fixture
def full_checkpoint(tmp_path):
    torch.set_num_threads(1)
    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=16, linear_value_head_dim=16,
            tie_word_embeddings=True,
            layer_types=['linear_attention', 'full_attention'],
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000,
                             'partial_rotary_factor': 1.0, 'mrope_section': [3, 3, 2]},
        ),
        vision_config=dict(
            depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
            out_hidden_size=32, patch_size=2, temporal_patch_size=1,
            spatial_merge_size=2, num_position_embeddings=16,
        ),
        image_token_id=120, video_token_id=121, vision_start_token_id=122,
        vision_end_token_id=123, pad_token_id=0, eos_token_id=2,
        tie_word_embeddings=True,
    )
    original = Qwen3_5ForConditionalGeneration(config).eval()
    original.save_pretrained(tmp_path)
    return original, tmp_path


def test_multimodal_checkpoint_loads_exact_language_weights_without_native_vision(full_checkpoint):
    original, path = full_checkpoint
    restored = _load_qwen35_language_model(path, range(128), device='cpu')
    assert isinstance(restored, Qwen3_5ForCausalLM)
    assert not hasattr(restored.model, 'visual')
    expected = {
        key.replace('model.language_model.', 'model.', 1): value
        for key, value in original.state_dict().items()
        if not key.startswith('model.visual.')
    }
    assert set(restored.state_dict()) == set(expected)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    assert restored.get_input_embeddings().num_embeddings == 128
    assert restored.lm_head.weight is restored.get_input_embeddings().weight


def test_text_loader_matches_cached_generation_scores_with_left_padding(full_checkpoint):
    original, path = full_checkpoint
    restored = _load_qwen35_language_model(path, range(128), device='cpu')
    ids = torch.tensor([[1, 4, 5], [0, 1, 5]])
    with torch.inference_mode():
        kwargs = dict(
            inputs_embeds=original.get_input_embeddings()(ids),
            attention_mask=torch.tensor([[1, 1, 1], [0, 1, 1]]),
            do_sample=False, max_new_tokens=4, eos_token_id=[], pad_token_id=0,
            return_dict_in_generate=True, output_scores=True, use_cache=True,
        )
        expected = original.generate(**kwargs)
        actual = restored.generate(**kwargs)
    assert torch.equal(actual.sequences, expected.sequences)
    assert len(actual.scores) == 4
    for actual_score, expected_score in zip(actual.scores, expected.scores):
        torch.testing.assert_close(actual_score, expected_score, rtol=0, atol=0)


def test_text_loader_rejects_missing_pretrained_decoder_weight(full_checkpoint):
    _, path = full_checkpoint
    checkpoint = path / 'model.safetensors'
    state = {key: value.clone() for key, value in load_file(checkpoint).items()}
    del state['model.language_model.layers.0.mlp.gate_proj.weight']
    save_file(state, checkpoint, metadata={'format': 'pt'})
    with pytest.raises(RuntimeError, match='Incomplete Qwen3.5 checkpoint'):
        _load_qwen35_language_model(path, range(128), device='cpu')
