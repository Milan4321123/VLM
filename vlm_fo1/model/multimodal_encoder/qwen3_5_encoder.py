import torch
import torch.nn as nn

from torchvision.transforms import ToPILImage

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel


class Qwen3_5VisionTower(nn.Module):
    """
    Vision backbone wrapper for Qwen3.5.

    Same interface as Qwen2_5_VlVisionTower but with much less machinery: the
    Qwen3.5 ViT runs full attention in every block and keeps tokens in raster
    order end to end, so the multi-level features needed for region encoding
    come from plain forward hooks instead of the monkey patch the Qwen2.5
    wrapper needs (no window partitioning, nothing to un-shuffle).
    """

    NUM_FEATURE_LEVELS = 4

    def __init__(self, image_tower, args, delay_load=False, min_pixels=56*56, max_pixels=2048*2048):
        super().__init__()

        self.is_loaded = False
        self.image_tower_name = image_tower

        self.use_vision_tower_region_feature = getattr(args, 'mm_use_vision_tower_region_feature', False)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.delay_load = delay_load
        self._level_features = []

        self.cfg_only = args.vision_config
        self.load_model(model_path=getattr(args, '_name_or_path', None) or getattr(args, 'name_or_path', None))

    def load_model(self, model_path=None, image_size=336, is_train=True):
        """
        Build the Qwen3.5 vision transformer from config. The weights arrive
        through the outer model's from_pretrained (the tower is a registered
        submodule), the same way the Qwen2.5 wrapper works.
        """
        self.image_tower = Qwen3_5VisionModel._from_config(self.cfg_only)
        self.image_processor = self._build_image_processor(model_path)

        if self.use_vision_tower_region_feature:
            self._install_level_hooks()
        self.is_loaded = True

    def _build_image_processor(self, model_path):
        from transformers import AutoImageProcessor
        if model_path:
            try:
                return AutoImageProcessor.from_pretrained(
                    model_path, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
            except (TypeError, OSError, ValueError):
                pass
        # No processor files available (e.g. a from-scratch config): build one
        # from the vision config alone. The patching scheme (resize to patch
        # multiples, stack temporal pairs, flatten) is shared with Qwen2-VL.
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        return Qwen2VLImageProcessor(
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            patch_size=self.cfg_only.patch_size,
            temporal_patch_size=self.cfg_only.temporal_patch_size,
            merge_size=self.cfg_only.spatial_merge_size,
        )

    def _install_level_hooks(self):
        """
        Register forward hooks on four evenly spaced transformer blocks. Each
        hook stores that block's hidden states ([seq_len, hidden_size], raster
        order) so a forward pass leaves a 4-level feature pyramid behind.
        """
        depth = len(self.image_tower.blocks)
        self.level_ids = sorted({max(round(depth * k / self.NUM_FEATURE_LEVELS) - 1, 0)
                                 for k in range(1, self.NUM_FEATURE_LEVELS + 1)})
        for i, blk in enumerate(self.image_tower.blocks):
            if i in self.level_ids:
                blk.register_forward_hook(self._keep_level_output)

    def _keep_level_output(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self._level_features.append(hidden)

    def _collect_level_maps(self, image_grid_thw):
        """
        Turn the hooked [seq_len, C] tensors of the last image into spatial
        maps [1, C, grid_h, grid_w] for ROIAlign. Tokens are already in raster
        order, so this is a plain reshape.
        """
        t, grid_h, grid_w = [int(x) for x in image_grid_thw[0]]
        assert t == 1, "region features are only supported for single-frame images"
        maps = []
        for feat in self._level_features:
            fmap = feat.reshape(grid_h, grid_w, -1).permute(2, 0, 1).unsqueeze(0)
            maps.append(fmap)
        return maps

    def convert_image_format(self, image):
        """
        Convert raw image tensor to pre-processed model input tensor and grid shape.
        """
        pil_image = ToPILImage()(image)
        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        return inputs['pixel_values'], inputs['image_grid_thw']

    def forward(self, images, image_grid_thws=[]):
        """
        Forward pass for a batch (list) of images.
        Returns image features, grid thws, and optional multi-level features per image.
        """
        if type(images) is not list:
            raise NotImplementedError("Qwen3_5VisionTower only supports list-of-image input")

        image_features = []
        multi_level_features_list = []
        output_image_grid_thws = []

        for i, image in enumerate(images):
            if image_grid_thws is None or len(image_grid_thws) == 0:
                image, image_grid_thw = self.convert_image_format(image=image)
            else:
                image_grid_thw = image_grid_thws[i]

            self._level_features = []
            image_forward_out = self.image_tower(
                image.to(device=self.device, dtype=self.dtype),
                grid_thw=image_grid_thw.to(device=self.device),
            )
            # the Qwen3.5 tower returns pre-merge patch features as
            # last_hidden_state and the merged tokens as pooler_output
            merged = image_forward_out.pooler_output if hasattr(image_forward_out, 'pooler_output') \
                else image_forward_out
            image_features.append(merged.unsqueeze(0).to(self.dtype))
            output_image_grid_thws.append(image_grid_thw)

            if self.use_vision_tower_region_feature:
                multi_level_features_list.append(self._collect_level_maps(image_grid_thw))

        return image_features, output_image_grid_thws, multi_level_features_list

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.image_tower.dtype

    @property
    def device(self):
        return self.image_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.image_tower.config
        return self.cfg_only

    @property
    def hidden_size(self):
        # width of the merged tokens fed to the LLM projector
        return self.config.out_hidden_size
