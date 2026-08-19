import itertools
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel, Qwen3_5ForCausalLM

from vlm_fo1.model.multimodal_encoder.qwen3_5_encoder import Qwen3_5VisionTower
from vlm_fo1.model.multimodal_encoder.builder import build_vision_tower, build_vision_tower_aux
from vlm_fo1.model.multimodal_projector.builder import build_vision_projector, build_vision_projector_aux
from vlm_fo1.model.multimodal_visual_prompt_encoder.hybrid_finegrained_region_encoder import HFREModule
from vlm_fo1.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_REGION_INDEX

from ..omchat_arch import OmChatMetaModel, OmChatMetaForCausalLM


# Custom config which extends Qwen3_5Config for the OmChat multimodal model.
# Qwen3.5 uses nested text/vision sub-configs and, unlike Qwen2.5-VL, the
# composite config does not mirror the text attributes at the top level - so
# the ones the FO1 modules read (projector shapes, pad ids, init range) are
# materialized here as flat attributes.
class OmChatQwen35Config(Qwen3_5Config):
    model_type = "omchat_qwen3_5"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        text_config = self.get_text_config()
        for key in ("hidden_size", "vocab_size", "initializer_range",
                    "bos_token_id", "eos_token_id", "pad_token_id"):
            if getattr(self, key, None) is None and getattr(text_config, key, None) is not None:
                setattr(self, key, getattr(text_config, key))


# Core model definition: the Qwen3.5 text decoder plus the FO1 modules.
# OmChatMetaModel.__init__ cannot be reused directly because it forwards a
# single config to the decoder, while Qwen3_5TextModel wants the nested text
# config and the FO1 builders want the composite one - so the module setup
# from omchat_arch is replicated here with the right config going to each.
class OmChatQwen35Model(OmChatMetaModel, Qwen3_5TextModel):
    config_class = OmChatQwen35Config

    def __init__(self, config):
        Qwen3_5TextModel.__init__(self, config.get_text_config())

        delay_load = getattr(config, 'delay_load', True)
        if getattr(config, "mm_vision_tower", None) is not None:
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.mm_projector = build_vision_projector(config)
        if getattr(config, "mm_vision_tower_aux", None) is not None:
            self.vision_tower_aux = build_vision_tower_aux(config, delay_load=delay_load)
            self.object_vp_extractor = HFREModule(
                roi_output_size=getattr(config, "mm_roi_output_size", 7),
                region_feature_dim=config.mm_region_hidden_size,
                apply_position_embedding=getattr(config, "mm_apply_position_embedding", True),
                pos_embedding_strategy=getattr(config, "mm_pos_embedding_strategy", "bbox_based"),
                use_vt_region_feature_only=getattr(config, "mm_use_vt_region_feature_only", False),
                use_vision_tower_region_feature=getattr(config, "mm_use_vision_tower_region_feature", False),
                region_feature_combination=getattr(config, "mm_region_feature_combination", "concat"),
                apply_region_layer_norm=getattr(config, "mm_apply_region_layer_norm", False),
                vision_tower_region_feature_dim=self.get_vision_tower().config.hidden_size * Qwen3_5VisionTower.NUM_FEATURE_LEVELS if not getattr(config, "mm_use_simpleFPN_for_vt", False) else 2048,
                vision_tower_spatial_scale=1/self.get_vision_tower().config.patch_size,
                use_simpleFPN_for_vt=getattr(config, "mm_use_simpleFPN_for_vt", False),
                aux_vision_tower_spatial_scale=0.25,
                aux_vision_tower_region_feature_dims=[256, 512, 1024, 2048],
            )
            self.mm_projector_aux = build_vision_projector_aux(config)


# Main class for the multimodal CausalLM. Inherits the text-only Qwen3.5 LM
# head; all vision handling is owned by the FO1 modules, exactly like the
# Qwen2.5 variant. The multimodal rope helpers (get_rope_index and
# get_vision_position_ids) are ported from transformers' Qwen3_5Model, which
# is not in this inheritance chain.
class OmChatQwen35ForCausalLM(Qwen3_5ForCausalLM, OmChatMetaForCausalLM):
    config_class = OmChatQwen35Config

    def __init__(self, config, delay_load=True):
        if not hasattr(config, 'delay_load'):
            config.delay_load = delay_load
        # skip Qwen3_5ForCausalLM.__init__ (it would build a plain text model
        # from this composite config); initialize the PreTrainedModel machinery
        # directly and attach our own model
        super(Qwen3_5ForCausalLM, self).__init__(config)
        self.model = OmChatQwen35Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        self.post_init()

    def get_model(self):
        return self.model

    # ------------------------------------------------------------------
    # multimodal rope, ported from transformers Qwen3_5Model
    # ------------------------------------------------------------------
    def get_vision_position_ids(self, start_position, grid_thw, temp_merge_size=1,
                                spatial_merge_size=1, time_interval=1, device=None):
        llm_grid_t, llm_grid_h, llm_grid_w = (
            int(grid_thw[0]) // temp_merge_size,
            int(grid_thw[1]) // spatial_merge_size,
            int(grid_thw[2]) // spatial_merge_size,
        )
        position_temporal = torch.arange(llm_grid_t, device=device) * time_interval
        position_height = torch.arange(llm_grid_h, device=device) + start_position
        position_width = torch.arange(llm_grid_w, device=device) + start_position
        t_grid, h_grid, w_grid = torch.meshgrid(position_temporal, position_height, position_width, indexing="ij")
        vision_position_ids = torch.stack([t_grid, h_grid, w_grid], dim=0).reshape(3, -1)
        vision_position_ids[0] += start_position
        return vision_position_ids

    def get_rope_index(self, input_ids, mm_token_type_ids, image_grid_thw=None,
                       video_grid_thw=None, attention_mask=None):
        """
        Compute (3, batch, seq) mrope position ids plus per-sample deltas.
        mm_token_type_ids: 0 = text, 1 = image, 2 = video.
        """
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision_config.spatial_merge_size

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3, input_ids.shape[0], input_ids.shape[1],
            dtype=input_ids.dtype, device=input_ids.device,
        )
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                input_type_group.append((key, group[0][0], group[-1][0] + 1))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                else:
                    grid_thw = next(grid_iters[modality_type])
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += max(int(grid_thw[1]), int(grid_thw[2])) // spatial_merge_size
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    # ------------------------------------------------------------------
    # image / region encoding (same flow as the Qwen2.5 variant)
    # ------------------------------------------------------------------
    def encode_images(self, images, images_grid_thw=None):
        if isinstance(self.get_model().get_vision_tower(), Qwen3_5VisionTower):
            image_features = self.get_model().get_vision_tower()(images, images_grid_thw)
            image_features, image_grid_thws, multi_level_features = image_features
            if type(image_features) is list:
                # List has items of shape (1, seq_len, dim)
                token_length_list = [i.shape[1] for i in image_features]
                image_features = torch.cat(image_features, dim=1)
        else:
            image_features = self.get_model().get_vision_tower()(images)
            image_grid_thws = None
            multi_level_features = None

        image_features = self.get_model().mm_projector(image_features)

        if isinstance(self.get_model().get_vision_tower(), Qwen3_5VisionTower):
            start = 0
            new_image_features = []
            for length in token_length_list:
                end = start + length
                new_image_features.append(image_features[:, start:end, :].squeeze(0))
                start = end
            image_features = new_image_features

        return image_features, image_grid_thws, multi_level_features

    def encode_regions(self, images, bbox_list, vt_multi_level_features=None, vt_images_size=None):
        aux_image_features_list = self.get_model().get_vision_tower_aux()(images)
        region_features = []
        if getattr(self.config, "mm_use_vision_tower_region_feature", False):
            image_features_list = vt_multi_level_features
            for batch_idx, (image_features, aux_image_features) in enumerate(zip(image_features_list, aux_image_features_list)):

                if getattr(self.config, "mm_use_simpleFPN_for_vt", False):
                    multilevel_visual_feats = image_features[-1]
                else:
                    multilevel_visual_feats = image_features
                multilevel_aux_visual_feats = aux_image_features["image_features"]
                boxes = bbox_list[batch_idx]

                if boxes is None or len(boxes) == 0:
                    boxes = torch.tensor([[0, 10, 0, 10]], device=multilevel_aux_visual_feats[0].device, dtype=torch.float32)

                boxes = boxes.to(torch.float32).to(multilevel_aux_visual_feats[0].device)
                current_image_height, current_image_width = images[batch_idx].shape[-2:]
                original_height, original_width = vt_images_size[batch_idx]
                scale_height = original_height / current_image_height
                scale_width = original_width / current_image_width
                vt_boxes = boxes * torch.tensor([scale_width, scale_height, scale_width, scale_height], device=boxes.device)

                extracted_region_feat = self.get_model().object_vp_extractor(
                    aux_multi_level_features=multilevel_aux_visual_feats,
                    vt_multi_level_features=multilevel_visual_feats,
                    aux_boxes=[boxes],
                    vt_boxes=[vt_boxes]
                ).squeeze(0).to(multilevel_aux_visual_feats[0].dtype)
                region_feat = self.get_model().mm_projector_aux(extracted_region_feat)  # [num_bbox, hidden]
                region_features.append(region_feat)
        else:
            for batch_idx, image_features in enumerate(aux_image_features_list):
                multilevel_visual_feats = image_features["image_features"]
                boxes = bbox_list[batch_idx]

                if boxes is None or len(boxes) == 0:
                    boxes = torch.tensor([[0, 10, 0, 10]], device=multilevel_visual_feats[0].device, dtype=torch.float32)

                multi_level_aux_features = multilevel_visual_feats
                boxes = boxes.to(torch.float32).to(multi_level_aux_features[0].device)
                extracted_region_feat = self.get_model().object_vp_extractor(
                    multi_level_aux_features,
                    [boxes],
                ).squeeze(0).to(multi_level_aux_features[0].dtype)
                region_feat = self.get_model().mm_projector_aux(extracted_region_feat)
                region_features.append(region_feat)

        return region_features

    # ------------------------------------------------------------------
    # embedding splice (ported from the Qwen2.5 variant)
    # ------------------------------------------------------------------
    def prepare_inputs_labels_for_qwen3_5_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, images_aux=None, bbox_list=None, image_grid_thws=None
    ):
        vision_tower = self.get_vision_tower()
        video_tower = self.get_video_tower()
        vision_tower_aux = self.get_vision_tower_aux()
        # Fast-path for non-multimodal case or decoding steps (single new token)
        if (vision_tower is None and video_tower is None) or images is None or input_ids.shape[1] == 1:
            cache_position = None
            if past_key_values is not None and (vision_tower is not None or video_tower is not None) and images is not None and input_ids.shape[1] == 1:
                if hasattr(past_key_values, "get_seq_length"):
                    target_shape = past_key_values.get_seq_length() + 1
                else:
                    target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)

                position_ids = None
                cache_position = torch.tensor([target_shape - 1], device=attention_mask.device)
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, cache_position

        # Indices for images (3D or 2D tensors) and videos (4D tensors)
        image_idx = [idx for idx, img in enumerate(images) if img.ndim == 3 or img.ndim == 2]
        video_idx = [idx for idx, vid in enumerate(images) if vid.ndim == 4]

        if isinstance(vision_tower, Qwen3_5VisionTower):
            images_minibatch = [images[idx] for idx in image_idx] if len(image_idx) > 0 else []
        else:
            images_minibatch = torch.stack([images[idx] for idx in image_idx]) if len(image_idx) > 0 else []
        videos_minibatch = torch.stack([images[idx] for idx in video_idx]) if len(video_idx) > 0 else []

        if vision_tower_aux is not None and images_aux is not None:
            images_minibatch_aux = [images_aux[idx].unsqueeze(0) for idx in image_idx] if len(image_idx) > 0 else []

        tmp_image_features = [None] * (len(image_idx) + len(video_idx))
        if getattr(images_minibatch, 'ndim', 0) == 4 or (type(images_minibatch) is list and len(images_minibatch) > 0):
            if vision_tower is not None:
                image_features_minibatch, image_grid_thws_minibatch, vt_multi_level_features_minibatch = self.encode_images(images_minibatch, image_grid_thws)
            else:
                image_features_minibatch = torch.randn(1).to(self.device)

            for i, pos in enumerate(image_idx):
                tmp_image_features[pos] = image_features_minibatch[i]

            if vision_tower_aux is not None and bbox_list is not None and len(bbox_list) > 0:
                if isinstance(self.get_model().get_vision_tower(), Qwen3_5VisionTower):
                    patch_size = self.get_model().get_vision_tower().config.patch_size
                    vt_images_size_minibatch = [im_grid_thw[0][-2:]*patch_size for im_grid_thw in image_grid_thws]
                    region_features = self.encode_regions(images_minibatch_aux, bbox_list, vt_multi_level_features_minibatch, vt_images_size_minibatch)
            else:
                region_features = None

        if getattr(videos_minibatch, 'ndim', 0) == 5:
            video_features_minibatch = self.encode_videos(videos_minibatch)
            for i, pos in enumerate(video_idx):
                tmp_image_features[pos] = video_features_minibatch[i]

        new_tmp = []
        for image in tmp_image_features:
            if isinstance(image, list):
                for i in range(len(image)):
                    new_tmp.append(image[i])
            else:
                new_tmp.append(image)
        image_features = new_tmp

        # =========================== Build multimodal input & target sequences =========================

        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        if vision_tower_aux is None and (bbox_list is None or all(x is None for x in bbox_list)):
            new_input_embeds = []
            new_labels = []
            new_input_ids = []
            cur_image_idx = 0
            image_nums_in_batch = []

            for batch_idx, cur_input_ids in enumerate(input_ids):
                num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
                image_nums_in_batch.append(num_images)
                if num_images == 0:
                    cur_image_features = image_features[cur_image_idx]
                    cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                    cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                    new_input_embeds.append(cur_input_embeds)
                    new_labels.append(labels[batch_idx])
                    new_input_ids.append(cur_input_ids)
                    cur_image_idx += 1
                    continue

                image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
                cur_input_ids_noim = []
                cur_labels = labels[batch_idx]
                cur_labels_noim = []
                for i in range(len(image_token_indices) - 1):
                    cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                    cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
                split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
                cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

                cur_new_input_embeds = []
                cur_new_labels = []
                cur_new_input_ids = []
                for i in range(num_images + 1):
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    cur_new_input_ids.append(cur_input_ids_noim[i])
                    if i < num_images:
                        cur_image_features = image_features[cur_image_idx].to(self.device)
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_input_ids.append(torch.full((cur_image_features.shape[0],), self.config.image_token_id, device=cur_labels.device, dtype=cur_labels.dtype))
                cur_new_input_embeds = torch.cat(cur_new_input_embeds)
                cur_new_labels = torch.cat(cur_new_labels)
                cur_new_input_ids = torch.cat(cur_new_input_ids)

                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)
                new_input_ids.append(cur_new_input_ids)
        else:
            new_input_embeds = []
            new_labels = []
            new_input_ids = []
            cur_image_idx = 0
            image_nums_in_batch = []

            for batch_idx, cur_input_ids in enumerate(input_ids):
                cur_region_idx = 0
                num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
                num_regions = (cur_input_ids == DEFAULT_REGION_INDEX).sum() if DEFAULT_REGION_INDEX in cur_input_ids else 0
                image_nums_in_batch.append(num_images)

                if num_images == 0 and num_regions == 0:
                    cur_image_features = image_features[cur_image_idx]
                    cur_region_features = region_features[cur_region_idx]
                    cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                    cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_region_features[0:0]], dim=0)
                    new_input_embeds.append(cur_input_embeds)
                    new_labels.append(labels[batch_idx])
                    new_input_ids.append(cur_input_ids)
                    cur_image_idx += 1
                    continue

                image_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                region_indices = torch.where(cur_input_ids == DEFAULT_REGION_INDEX)[0].tolist() if num_regions > 0 else []
                all_special_indices = sorted([-1] + image_indices + region_indices + [cur_input_ids.shape[0]])

                cur_input_ids_segments = []
                cur_labels = labels[batch_idx]
                cur_labels_segments = []

                for i in range(len(all_special_indices) - 1):
                    cur_input_ids_segments.append(cur_input_ids[all_special_indices[i]+1:all_special_indices[i+1]])
                    cur_labels_segments.append(cur_labels[all_special_indices[i]+1:all_special_indices[i+1]])

                split_sizes = [x.shape[0] for x in cur_labels_segments]
                cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_segments))
                if num_regions == 0 and vision_tower_aux is not None and region_features is not None:
                    cur_region_features = region_features[cur_region_idx]
                    temp_input_embeds = torch.cat([cur_input_embeds, cur_region_features[0:0]], dim=0)
                    cur_input_embeds = temp_input_embeds

                cur_input_embeds_segments = torch.split(cur_input_embeds, split_sizes, dim=0)

                cur_new_input_embeds = []
                cur_new_labels = []
                cur_new_input_ids = []

                for i in range(len(all_special_indices) - 1):
                    cur_new_input_embeds.append(cur_input_embeds_segments[i])
                    cur_new_labels.append(cur_labels_segments[i])
                    cur_new_input_ids.append(cur_input_ids_segments[i])
                    if all_special_indices[i+1] in image_indices:
                        cur_image_features = image_features[cur_image_idx].to(self.device)
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_input_ids.append(torch.full((cur_image_features.shape[0],), self.config.image_token_id, device=cur_labels.device, dtype=cur_labels.dtype))

                    elif all_special_indices[i+1] in region_indices:
                        cur_region_features = region_features[batch_idx][cur_region_idx].to(self.device).unsqueeze(0)
                        cur_region_idx += 1
                        cur_new_input_embeds.append(cur_region_features)

                        cur_new_labels.append(torch.full((cur_region_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_input_ids.append(torch.full((cur_region_features.shape[0],), DEFAULT_REGION_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                cur_new_input_embeds = torch.cat(cur_new_input_embeds)
                cur_new_labels = torch.cat(cur_new_labels)
                cur_new_input_ids = torch.cat(cur_new_input_ids)
                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)
                new_input_ids.append(cur_new_input_ids)

        # Truncate sequences to maximum model length, if image+region tokens caused overflow
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Pad sequences in the batch to same length; compute batch masks
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        pad_token = self.config.bos_token_id if getattr(self.config, 'bos_token_id', None) is not None else 0

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_input_ids_padded = torch.full((batch_size, max_len), pad_token, dtype=new_input_ids[0].dtype, device=new_input_ids[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels, cur_new_input_ids) in enumerate(zip(new_input_embeds, new_labels, new_input_ids)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    new_input_ids_padded[i, :cur_len] = cur_new_input_ids
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        new_input_ids = new_input_ids_padded

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # Compute mrope position ids from the spliced sequence
        if isinstance(self.get_model().get_vision_tower(), Qwen3_5VisionTower):
            image_grid_thws = []
            cur_image_idx = 0
            for num_images in image_nums_in_batch:
                if num_images == 0:
                    cur_image_idx += 1
                    continue
                image_grid_thws += image_grid_thws_minibatch[cur_image_idx:cur_image_idx+num_images]
                cur_image_idx += num_images

            if len(image_grid_thws) > 0:
                image_grid_thws = torch.cat(image_grid_thws, dim=0)
            else:
                image_grid_thws = None

            # 0 = text, 1 = image; region tokens count as text positions
            mm_token_type_ids = (new_input_ids == self.config.image_token_id).to(torch.int32)

            position_ids, rope_deltas = self.get_rope_index(
                new_input_ids,
                mm_token_type_ids,
                image_grid_thw=image_grid_thws,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
            cache_position = torch.arange(new_input_embeds.shape[1], device=new_input_embeds.device)
        else:
            rope_deltas = None
            cache_position = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, rope_deltas, cache_position

    # Patch forward() to route through the multimodal splice first
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        images_aux: Optional[torch.FloatTensor] = None,
        bbox_list: Optional[torch.FloatTensor] = None,
        image_grid_thws: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                rope_deltas,
                cache_position
            ) = self.prepare_inputs_labels_for_qwen3_5_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                images_aux,
                bbox_list,
                image_grid_thws
            )

        if rope_deltas is not None:
            self.rope_deltas = rope_deltas

        # Decoding steps arrive without position ids. The text-only parent
        # would count raw cache offsets, which drifts from the mrope positions
        # used at prefill - rebuild them from cache_position + stored deltas
        # (this is what Qwen3_5ForConditionalGeneration does internally).
        if position_ids is None and self.rope_deltas is not None and past_key_values is not None:
            ref = input_ids if input_ids is not None else inputs_embeds
            batch_size, seq_len = ref.shape[0], ref.shape[1]
            device = ref.device
            position_ids = torch.arange(seq_len, device=device).view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(device)
                position_ids = position_ids + delta
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        return out

    # Keep the extra multimodal kwargs flowing through generate()
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        images: Optional[torch.FloatTensor] = None,
        images_aux: Optional[torch.FloatTensor] = None,
        bbox_list: Optional[torch.FloatTensor] = None,
        image_grid_thws: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            images=images,
            images_aux=images_aux,
            bbox_list=bbox_list,
            image_grid_thws=image_grid_thws,
            **kwargs,
        )
        return model_inputs


AutoConfig.register("omchat_qwen3_5", OmChatQwen35Config)
AutoModelForCausalLM.register(OmChatQwen35Config, OmChatQwen35ForCausalLM)
