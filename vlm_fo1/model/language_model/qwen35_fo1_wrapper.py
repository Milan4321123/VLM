import warnings

import torch
import torch.nn as nn


class Qwen35FO1Wrapper(nn.Module):
    def __init__(self, vision_backbone, language_model):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.language_model = language_model
        self.config = vision_backbone.config

        vision_width = int(self.config.hidden_size)
        language_width = self._language_hidden_size(language_model)

        if vision_width == language_width:
            self.feature_bridge = nn.Identity()
        else:
            self.feature_bridge = nn.Linear(
                vision_width,
                language_width,
                bias=False,
            )
            warnings.warn(
                f"The {vision_width}->{language_width} feature bridge is "
                "randomly initialized. It is enough to test the connection, "
                "but it is not trained for grounding.",
                RuntimeWarning,
            )

        self.feature_bridge.to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _language_hidden_size(language_model):
        config = language_model.config
        if hasattr(config, "text_config"):
            return int(config.text_config.hidden_size)
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        return int(language_model.get_input_embeddings().embedding_dim)

    @property
    def device(self):
        return self.language_model.get_input_embeddings().weight.device

    @property
    def dtype(self):
        return self.language_model.get_input_embeddings().weight.dtype

    def get_vision_tower(self):
        return self.vision_backbone.get_vision_tower()

    def get_vision_tower_aux(self):
        return self.vision_backbone.get_vision_tower_aux()

    def _bridge(self, features):
        features = features.to(device=self.device, dtype=self.dtype)
        return self.feature_bridge(features)

    def encode_visual_features(
        self,
        images,
        images_aux=None,
        bbox_list=None,
        image_grid_thws=None,
    ):
        if images is None or len(images) == 0:
            return [], None

        image_features, output_grids, multi_level_features = (
            self.vision_backbone.encode_images(images, image_grid_thws)
        )
        image_features = [self._bridge(features) for features in image_features]

        region_features = None
        if (
            self.get_vision_tower_aux() is not None
            and images_aux is not None
            and bbox_list is not None
            and len(bbox_list) > 0
        ):
            auxiliary_batch = [image.unsqueeze(0) for image in images_aux]
            patch_size = self.get_vision_tower().config.patch_size
            grids = image_grid_thws if image_grid_thws is not None else output_grids
            image_sizes = [grid[0][-2:] * patch_size for grid in grids]

            region_features = self.vision_backbone.encode_regions(
                auxiliary_batch,
                bbox_list,
                multi_level_features,
                image_sizes,
            )
            region_features = [
                self._bridge(features) for features in region_features
            ]

        return image_features, region_features
