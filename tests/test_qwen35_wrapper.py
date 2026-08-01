import unittest
import warnings
from types import SimpleNamespace

import torch

from vlm_fo1.model.language_model.qwen35_fo1_wrapper import Qwen35FO1Wrapper
from vlm_fo1.constants import DEFAULT_REGION_INDEX, IMAGE_TOKEN_INDEX


class FakeLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(100, 8)
        self.config = SimpleNamespace(hidden_size=8)

    def get_input_embeddings(self):
        return self.embeddings

    def generate(self, **kwargs):
        self.last_inputs = kwargs
        return torch.tensor([[1, 42, 43]])


class FakeVisionTower:
    config = SimpleNamespace(patch_size=2)


class FakeVisionBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)

    def get_vision_tower(self):
        return FakeVisionTower()

    def get_vision_tower_aux(self):
        return object()

    def encode_images(self, images, grids):
        return [torch.ones(3, 16)], grids, ["multi-level features"]

    def encode_regions(self, images, boxes, multi_level, image_sizes):
        return [torch.ones(2, 16)]


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    bos_token_id = 1


class Qwen35WrapperTest(unittest.TestCase):
    def setUp(self):
        self.language_model = FakeLanguageModel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.wrapper = Qwen35FO1Wrapper(
                FakeVisionBackbone(),
                self.language_model,
                FakeTokenizer(),
            )

    def test_image_and_region_features_are_projected(self):
        images, regions = self.wrapper.encode_visual_features(
            [torch.ones(3, 4, 4)],
            images_aux=[torch.ones(3, 4, 4)],
            bbox_list=[torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3]])],
            image_grid_thws=[torch.tensor([[1, 2, 2]])],
        )

        self.assertEqual(images[0].shape, (3, 8))
        self.assertEqual(regions[0].shape, (2, 8))
        self.assertEqual(self.wrapper.feature_bridge.weight.shape, (8, 16))

    def test_generate_replaces_visual_markers(self):
        input_ids = torch.tensor(
            [[10, IMAGE_TOKEN_INDEX, 11, DEFAULT_REGION_INDEX, 12,
              DEFAULT_REGION_INDEX, 13]]
        )
        output = self.wrapper.generate(
            inputs=input_ids,
            images=[torch.ones(3, 4, 4)],
            images_aux=[torch.ones(3, 4, 4)],
            bbox_list=[torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3]])],
            image_grid_thws=[torch.tensor([[1, 2, 2]])],
        )

        model_inputs = self.language_model.last_inputs
        self.assertEqual(model_inputs["inputs_embeds"].shape, (1, 9, 8))
        self.assertEqual(model_inputs["attention_mask"].tolist(), [[1] * 9])
        self.assertEqual(
            output.tolist(),
            [[10, IMAGE_TOKEN_INDEX, 11, DEFAULT_REGION_INDEX, 12,
              DEFAULT_REGION_INDEX, 13, 42, 43]],
        )


if __name__ == "__main__":
    unittest.main()
