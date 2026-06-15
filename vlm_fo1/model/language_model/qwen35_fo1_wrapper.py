import warnings
from typing import Optional

import torch
import torch.nn as nn

from vlm_fo1.constants import IMAGE_TOKEN_INDEX, DEFAULT_REGION_INDEX


class Qwen35FO1InferenceWrapper(nn.Module):
    """
    Inference-only bridge from the existing VLM-FO1 visual stack to Qwen3.5.

    VLM-FO1 was trained around Qwen2.5-VL's 2048-wide token embedding space.  The
    requested Qwen/Qwen3.5-0.8B model uses a 1024-wide text embedding space, so
    visual and region features must be projected before they can be spliced into
    Qwen3.5's input embeddings.
    """

    def __init__(
        self,
        vision_backbone: nn.Module,
        language_model: nn.Module,
        tokenizer,
        bridge_state_path: Optional[str] = None,
    ):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.config = vision_backbone.config

        self.language_hidden_size = self._get_language_hidden_size(language_model)
        self.vision_hidden_size = int(getattr(self.config, "hidden_size"))

        if self.vision_hidden_size == self.language_hidden_size:
            self.feature_bridge = nn.Identity()
            self.bridge_is_trained = True
        else:
            # Old behavior: VLM-FO1 projectors produced Qwen2.5-VL-sized 2048
            # embeddings. New behavior: Qwen3.5 consumes 1024-wide embeddings,
            # so this bridge is required for shape correctness. It must be
            # trained or loaded from a conversion checkpoint for meaningful VLM
            # quality.
            self.feature_bridge = nn.Linear(self.vision_hidden_size, self.language_hidden_size, bias=False)
            self.bridge_is_trained = False

        if bridge_state_path:
            state = torch.load(bridge_state_path, map_location="cpu")
            self.feature_bridge.load_state_dict(state)
            self.bridge_is_trained = True
        elif not self.bridge_is_trained:
            warnings.warn(
                "Qwen3.5 hidden size is different from the original VLM-FO1 "
                "Qwen2.5-VL hidden size. A new 2048->1024 feature bridge was "
                "initialized randomly. Inference is dimensionally valid, but "
                "the bridge/projectors need retraining or a converted checkpoint "
                "for reliable answers.",
                RuntimeWarning,
            )

        self.language_model_name_or_path = getattr(language_model.config, "_name_or_path", "Qwen/Qwen3.5-0.8B")

    @staticmethod
    def _get_language_hidden_size(language_model: nn.Module) -> int:
        config = language_model.config
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            return int(config.text_config.hidden_size)
        embeddings = language_model.get_input_embeddings()
        return int(embeddings.embedding_dim)

    @property
    def device(self):
        return self.language_model.get_input_embeddings().weight.device

    @property
    def dtype(self):
        return self.language_model.get_input_embeddings().weight.dtype

    def eval(self):
        self.vision_backbone.eval()
        self.language_model.eval()
        return super().eval()

    def get_vision_tower(self):
        return self.vision_backbone.get_vision_tower()

    def get_vision_tower_aux(self):
        return self.vision_backbone.get_vision_tower_aux()

    def _bridge_features(self, features: torch.Tensor) -> torch.Tensor:
        self.feature_bridge.to(device=self.device, dtype=self.dtype)
        return self.feature_bridge(features.to(device=self.device, dtype=self.dtype))

    def _embed_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(self.device)
        return self.language_model.get_input_embeddings()(token_ids)

    def _safe_prefix_ids(self, length: int, fill_token_id: int, device) -> torch.Tensor:
        return torch.full((length,), fill_token_id, dtype=torch.long, device=device)

    def _image_token_id(self, fallback_token_id: int) -> int:
        token_id = getattr(self.language_model.config, "image_token_id", None)
        if token_id is not None:
            return int(token_id)
        token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            return fallback_token_id
        return int(token_id)

    def _encode_visual_features(self, images, images_aux=None, bbox_list=None, image_grid_thws=None):
        if images is None or len(images) == 0:
            return [], None

        image_features, image_grid_thws_out, vt_multi_level_features = self.vision_backbone.encode_images(
            images,
            image_grid_thws,
        )
        image_features = [self._bridge_features(feature) for feature in image_features]

        region_features = None
        vision_tower_aux = self.vision_backbone.get_vision_tower_aux()
        if vision_tower_aux is not None and images_aux is not None and bbox_list is not None and len(bbox_list) > 0:
            images_aux_batch = [image_aux.unsqueeze(0) for image_aux in images_aux]
            patch_size = self.vision_backbone.get_vision_tower().config.patch_size
            grid_source = image_grid_thws if image_grid_thws is not None else image_grid_thws_out
            vt_images_size = [grid[0][-2:] * patch_size for grid in grid_source]
            region_features = self.vision_backbone.encode_regions(
                images_aux_batch,
                bbox_list,
                vt_multi_level_features,
                vt_images_size,
            )
            region_features = [self._bridge_features(feature) for feature in region_features]

        return image_features, region_features

    def _prepare_qwen_inputs(
        self,
        input_ids: torch.LongTensor,
        images=None,
        images_aux=None,
        bbox_list=None,
        image_grid_thws=None,
        attention_mask=None,
        pad_token_id=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        image_token_id = self._image_token_id(pad_token_id)

        image_features, region_features = self._encode_visual_features(
            images,
            images_aux=images_aux,
            bbox_list=bbox_list,
            image_grid_thws=image_grid_thws,
        )

        prompt_embeds = []
        prompt_ids_for_return = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_input_ids = cur_input_ids[attention_mask[batch_idx].to(cur_input_ids.device)]
            special_positions = torch.where(
                (cur_input_ids == IMAGE_TOKEN_INDEX) | (cur_input_ids == DEFAULT_REGION_INDEX)
            )[0].tolist()
            boundaries = [-1] + special_positions + [cur_input_ids.shape[0]]

            cur_embeds = []
            cur_return_ids = []
            cur_region_idx = 0

            for idx in range(len(boundaries) - 1):
                text_ids = cur_input_ids[boundaries[idx] + 1:boundaries[idx + 1]]
                if text_ids.numel() > 0:
                    cur_embeds.append(self._embed_text(text_ids))
                    cur_return_ids.append(text_ids.to(self.device))

                if idx == len(boundaries) - 2:
                    continue

                special_token = cur_input_ids[boundaries[idx + 1]].item()
                if special_token == IMAGE_TOKEN_INDEX:
                    if cur_image_idx >= len(image_features):
                        raise ValueError("Prompt contains more <image> markers than prepared images.")
                    feature = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_embeds.append(feature)
                    cur_return_ids.append(self._safe_prefix_ids(feature.shape[0], image_token_id, self.device))
                elif special_token == DEFAULT_REGION_INDEX:
                    if region_features is None:
                        raise ValueError("Prompt contains <regionfeat> markers but no region features were prepared.")
                    feature = region_features[batch_idx][cur_region_idx].unsqueeze(0)
                    cur_region_idx += 1
                    cur_embeds.append(feature)
                    cur_return_ids.append(self._safe_prefix_ids(feature.shape[0], pad_token_id, self.device))

            prompt_embeds.append(torch.cat(cur_embeds, dim=0))
            prompt_ids_for_return.append(torch.cat(cur_return_ids, dim=0))

        max_len = max(embed.shape[0] for embed in prompt_embeds)
        batch_size = len(prompt_embeds)
        padded_embeds = []
        padded_return_ids = torch.full(
            (batch_size, max_len),
            pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        padded_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)

        for idx, (embed, return_ids) in enumerate(zip(prompt_embeds, prompt_ids_for_return)):
            length = embed.shape[0]
            padded_embeds.append(
                torch.cat(
                    [
                        embed,
                        torch.zeros(
                            (max_len - length, embed.shape[1]),
                            dtype=embed.dtype,
                            device=embed.device,
                        ),
                    ],
                    dim=0,
                )
            )
            padded_return_ids[idx, :length] = return_ids
            padded_attention_mask[idx, :length] = 1

        return torch.stack(padded_embeds, dim=0), padded_attention_mask, padded_return_ids

    def generate(
        self,
        inputs=None,
        input_ids=None,
        images=None,
        images_aux=None,
        image_grid_thws=None,
        bbox_list=None,
        attention_mask=None,
        max_new_tokens=512,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=None,
        streamer=None,
        stopping_criteria=None,
        use_cache=True,
        **kwargs,
    ):
        if input_ids is None:
            input_ids = inputs
        if input_ids is None:
            raise ValueError("Qwen35FO1InferenceWrapper.generate requires input_ids or inputs.")

        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        inputs_embeds, qwen_attention_mask, _ = self._prepare_qwen_inputs(
            input_ids,
            images=images,
            images_aux=images_aux,
            bbox_list=bbox_list,
            image_grid_thws=image_grid_thws,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
        )

        if stopping_criteria is not None:
            # The old stopping criteria was built against full prompt token ids.
            # Qwen3.5 is called with inputs_embeds, so EOS ids are the stable
            # stopping mechanism here.
            warnings.warn(
                "Ignoring keyword stopping criteria for Qwen3.5 inputs_embeds generation; "
                "using eos_token_id instead.",
                RuntimeWarning,
            )

        generation_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=qwen_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            use_cache=use_cache,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            streamer=streamer,
        )
        if do_sample:
            generation_kwargs["temperature"] = temperature

        generation_kwargs.update(kwargs)
        generated_ids = self.language_model.generate(**generation_kwargs)

        # HF generation with inputs_embeds commonly returns only a BOS/pad seed
        # plus new tokens. Existing VLM-FO1 callers expect the original prompt
        # ids plus completion and slice by generation_kwargs["inputs"].shape[1].
        # Reattach the unexpanded prompt ids, not the internal visual-token
        # sequence, so that old decode logic still slices at the right place.
        seed_ids = {pad_token_id}
        if self.tokenizer.bos_token_id is not None:
            seed_ids.add(self.tokenizer.bos_token_id)
        seed_tensor = torch.tensor(list(seed_ids), device=generated_ids.device, dtype=generated_ids.dtype)
        if generated_ids.shape[1] > 1 and torch.isin(generated_ids[:, :1], seed_tensor).all():
            generated_ids = generated_ids[:, 1:]
        return torch.cat([input_ids.to(generated_ids.device), generated_ids], dim=1)
