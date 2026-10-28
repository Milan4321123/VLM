"""Offline mechanics checks. Tiny random models do not measure grounding quality."""
import tempfile
import unittest
import warnings
from base64 import b64encode
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, StoppingCriteria

from vlm_fo1.constants import DEFAULT_REGION_INDEX, IMAGE_TOKEN_INDEX
from vlm_fo1.mm_utils import load_image, prepare_inputs, resize_shortest_edge_images_and_bboxes
from vlm_fo1.model.language_model.qwen35_fo1_wrapper import Qwen35FO1InferenceWrapper


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = None
    unk_token_id = None
    eos_token = '<|im_end|>'
    name_or_path = 'Qwen/Qwen3.5-0.8B'
    chat_template = 'enable_thinking'

    def encode(self, text, **kwargs):
        return [10 + ord(char) % 100 for char in text]

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=self.encode(text))

    def convert_tokens_to_ids(self, token):
        return {'<|im_start|>': 1, '<|im_end|>': 2}.get(token)

    def batch_decode(self, ids, **kwargs):
        return ['' for _ in ids]


class Vision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=64, model_type='omchat_qwen2_5_vl',
                                      mm_use_region_index_token=True)
        self.tower = SimpleNamespace(config=SimpleNamespace(patch_size=14))

    def get_vision_tower(self):
        return self.tower

    def get_vision_tower_aux(self):
        return self.tower

    def encode_images(self, images, grids):
        return [torch.ones(4, 64) for _ in images], grids, [None for _ in images]

    def encode_regions(self, images, boxes, features, sizes):
        return [torch.ones(len(box), 64) for box in boxes]


class Processor:
    def preprocess(self, image, **kwargs):
        return {'pixel_values': torch.zeros(1, 3, 28, 28), 'image_grid_thw': torch.tensor([[1, 2, 2]])}


def tiny_language_model():
    config = Qwen3_5Config(
        text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
                         linear_key_head_dim=16, linear_value_head_dim=16,
                         layer_types=['linear_attention', 'full_attention'],
                         rope_parameters={'rope_type': 'default', 'rope_theta': 10000,
                                          'partial_rotary_factor': 1.0, 'mrope_section': [3, 3, 2]}),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=32, patch_size=2, temporal_patch_size=1,
                           spatial_merge_size=2, num_position_embeddings=16),
        image_token_id=120, video_token_id=121, vision_start_token_id=122,
        vision_end_token_id=123, pad_token_id=0, eos_token_id=2,
    )
    return Qwen3_5ForConditionalGeneration(config).eval()


class MechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.language_model = tiny_language_model()

    def wrapper(self, seed=0):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return Qwen35FO1InferenceWrapper(Vision(), self.language_model, Tokenizer(), bridge_seed=seed)

    def visual_inputs(self):
        return dict(input_ids=torch.tensor([[1, IMAGE_TOKEN_INDEX, 4, DEFAULT_REGION_INDEX, 5]]),
                    images=[torch.zeros(3, 28, 28)], images_aux=[torch.zeros(3, 28, 28)],
                    bbox_list=[torch.tensor([[0, 0, 10, 10]])],
                    image_grid_thws=[torch.tensor([[1, 2, 2]])])

    def test_bridge_reproducible_frozen_and_rng_preserved(self):
        state = torch.random.get_rng_state().clone()
        first, second = self.wrapper(), self.wrapper()
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(first.feature_bridge.weight, second.feature_bridge.weight))
        self.assertFalse(any(p.requires_grad for p in first.parameters()))
        self.assertFalse(first.inference_status['fine_tuning_performed'])
        self.assertFalse(first.inference_status['grounding_quality_validated'])

    def test_real_qwen_hybrid_cache_and_prompt_prefix(self):
        wrapper = self.wrapper()
        inputs = self.visual_inputs()
        output = wrapper.generate(**inputs, max_new_tokens=3, eos_token_id=[], use_cache=True)
        self.assertEqual(tuple(output.shape), (1, inputs['input_ids'].shape[1] + 3))
        self.assertTrue(torch.equal(output[:, :5], inputs['input_ids']))

    def test_nonfinite_visual_features_fail_before_generation(self):
        wrapper = self.wrapper()
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, 'visual encoder.*non-finite'):
                wrapper._bridge_features(torch.full((4, 64), value))
        with torch.no_grad():
            wrapper.feature_bridge.weight.fill_(float('inf'))
        with self.assertRaisesRegex(RuntimeError, 'feature bridge.*non-finite'):
            wrapper._bridge_features(torch.ones(4, 64))

    def test_caller_stopping_criteria_sees_original_prompt(self):
        lengths = []

        class StopAfterOne(StoppingCriteria):
            def __call__(self, ids, scores, **kwargs):
                lengths.append(ids.shape[1])
                return torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)

        output = self.wrapper().generate(**self.visual_inputs(), max_new_tokens=4,
                                         stopping_criteria=[StopAfterOne()], return_dict_in_generate=True)
        self.assertEqual(lengths, [6])
        self.assertEqual(output.sequences.shape[1], 6)

    def test_missing_regions_fail_clearly(self):
        inputs = self.visual_inputs()
        inputs['bbox_list'] = [torch.empty(0, 4)]
        with self.assertRaisesRegex(ValueError, 'more <regionfeat>'):
            self.wrapper().generate(**inputs)

    def test_left_padding_text_batch(self):
        embeds, mask = self.wrapper()._prepare_qwen_inputs(
            torch.tensor([[1, 4, 5], [0, 0, 7]]), attention_mask=torch.tensor([[1, 1, 1], [0, 0, 1]]))
        self.assertEqual(mask.tolist(), [[1, 1, 1], [0, 0, 1]])
        self.assertEqual(tuple(embeds.shape), (2, 3, 32))
        self.assertEqual(embeds[1, :2].count_nonzero().item(), 0)

    def test_image_only_request_and_history(self):
        image = Image.new('RGB', (64, 32))
        messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image}}]},
                    {'role': 'assistant', 'content': 'I see an image.'},
                    {'role': 'user', 'content': 'Describe the system in it.'}]
        args = prepare_inputs('local-model', Vision(), (Processor(), Processor()), Tokenizer(), messages,
                              device='cpu', stream=False, verbose=False)
        self.assertEqual(len(args['images']), 1)
        self.assertEqual((args['inputs'] == IMAGE_TOKEN_INDEX).sum().item(), 1)
        self.assertIsNone(args['bbox_list'])

    def test_too_many_regions_rejected_before_tokenization(self):
        message = {'role': 'user', 'content': 'test', 'bbox_list': [[0, 0, 1, 1]] * 101}
        with self.assertRaisesRegex(ValueError, 'at most 100'):
            prepare_inputs('local-model', Vision(), (None, None), Tokenizer(), [message], device='cpu')

    def test_resize_keeps_boxes_and_accepts_empty(self):
        image = Image.new('RGB', (100, 50))
        images, boxes = resize_shortest_edge_images_and_bboxes([image], [[10, 10, 20, 20]], [100])
        self.assertEqual(images[0].size, (200, 100))
        self.assertEqual(boxes, [[20, 20, 40, 40]])
        _, empty = resize_shortest_edge_images_and_bboxes([image], [])
        self.assertEqual(empty, [])

    def test_load_image_preserves_size_for_paired_box_resize(self):
        image = Image.new('RGB', (14, 7))
        encoded = BytesIO()
        image.save(encoded, format='JPEG')
        content = encoded.getvalue()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'small.jpg'
            path.write_bytes(content)
            sources = [image, str(path), 'data:image/jpeg;base64,' + b64encode(content).decode(),
                       'https://example.test/small.jpg']
            with patch('vlm_fo1.mm_utils.requests.get', return_value=SimpleNamespace(content=content)):
                for source in sources:
                    with self.subTest(source_type=type(source).__name__):
                        self.assertEqual(load_image(source).size, image.size)

    def test_prepare_inputs_scales_small_image_boxes_with_image(self):
        for size, box, expected in [
            ((14, 14), [0, 0, 14, 14], [0, 0, 28, 28]),
            ((64, 32), [16, 8, 48, 24], [7, 7, 21, 21]),
        ]:
            with self.subTest(size=size):
                messages = [{'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': Image.new('RGB', size)}},
                    {'type': 'text', 'text': 'Select this region.'},
                ], 'bbox_list': [box]}]
                args = prepare_inputs('local-model', Vision(), (Processor(), Processor()), Tokenizer(),
                                      messages, device='cpu', stream=False, verbose=False)
                self.assertEqual(tuple(args['images_aux'][0].shape[-2:]), (28, 28))
                torch.testing.assert_close(args['bbox_list'][0], torch.tensor([expected], dtype=torch.float32))
                self.assertEqual(messages[0]['bbox_list'], [box])


if __name__ == '__main__':
    unittest.main()
