import warnings
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
from transformers import StoppingCriteria

from vlm_fo1.constants import IMAGE_TOKEN_INDEX, DEFAULT_REGION_INDEX


class _PromptAwareStoppingCriteria(StoppingCriteria):
    """Keep callers' prompt-length based criteria valid with inputs_embeds."""

    def __init__(self, criterion, prompt_ids):
        self.criterion = criterion
        self.prompt_ids = prompt_ids

    def __call__(self, input_ids, scores, **kwargs):
        repeats = input_ids.shape[0] // self.prompt_ids.shape[0]
        prompt = self.prompt_ids.repeat_interleave(repeats, dim=0).to(input_ids.device)
        return self.criterion(torch.cat([prompt, input_ids], dim=1), scores, **kwargs)


class Qwen35FO1InferenceWrapper(nn.Module):
    """
    Inference-only bridge from the existing VLM-FO1 visual stack to Qwen3.5.

    VLM-FO1 was trained around Qwen2.5-VL's 2048-wide token embedding space,
    while Qwen3.5-0.8B embeds text in 1024 dims, so the visual and region
    features have to be projected before they can be spliced into Qwen3.5's
    input embeddings. Generation then runs entirely through Qwen3.5 via
    inputs_embeds; the FO1 checkpoint's own decoder is never called.
    """

    def __init__(
        self,
        vision_backbone: nn.Module,
        language_model: nn.Module,
        tokenizer,
        bridge_state_path: Optional[str] = None,
        bridge_seed: int = 0,
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
        else:
            # The FO1 projectors emit features sized for the decoder they were
            # trained with (2048 for Qwen2.5-VL); Qwen3.5-0.8B embeds text in
            # 1024 dims, so a projection is needed before the concat is even
            # shape-valid. Without a trained state this layer is random and the
            # answers will not be grounded — see the warning below.
            # A fixed CPU seed makes the no-training experiment reproducible
            # without changing the application's RNG state.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(bridge_seed)
                self.feature_bridge = nn.Linear(self.vision_hidden_size, self.language_hidden_size, bias=False)

        self.bridge_is_trained = False
        if bridge_state_path:
            state = torch.load(bridge_state_path, map_location="cpu", weights_only=True)
            self.feature_bridge.load_state_dict(state)
        else:
            warnings.warn(
                "The FO1-to-Qwen3.5 feature alignment and added region tokens "
                "are not trained. This inference-only migration checks execution; "
                "its outputs do not establish reliable visual grounding.",
                RuntimeWarning,
            )

        self.language_model_name_or_path = getattr(language_model.config, "_name_or_path", "Qwen/Qwen3.5-0.8B")
        self.inference_status = {
            "backend": "fo1_hfre_qwen35",
            "fine_tuning_performed": False,
            "bridge_seed": bridge_seed,
            "bridge_state_loaded": bool(bridge_state_path),
            "bridge_alignment": "external_state_unverified" if bridge_state_path else "untrained",
            "region_token_alignment": "untrained",
            "grounding_quality_validated": False,
        }
        self.requires_grad_(False)
        self.eval()

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
        if not torch.isfinite(features).all():
            raise RuntimeError("The visual encoder produced non-finite features; inference cannot continue.")
        self.feature_bridge.to(device=self.device, dtype=self.dtype)
        projected = self.feature_bridge(features.to(device=self.device, dtype=self.dtype))
        if not torch.isfinite(projected).all():
            raise RuntimeError("The visual feature bridge produced non-finite embeddings; inference cannot continue.")
        return projected

    def _embed_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(self.device)
        return self.language_model.get_input_embeddings()(token_ids)

    def _encode_visual_features(self, images, images_aux=None, bbox_list=None, image_grid_thws=None):
        if images is None or len(images) == 0:
            return [], None

        started = time.perf_counter()
        print("[VLM-FO1] Analysing the image...", file=sys.stderr, flush=True)
        image_features, image_grid_thws_out, vt_multi_level_features = self.vision_backbone.encode_images(
            images,
            image_grid_thws,
        )
        image_features = [self._bridge_features(feature) for feature in image_features]
        print(f"[VLM-FO1] Image features ready ({time.perf_counter() - started:.1f}s).", file=sys.stderr, flush=True)

        region_features = None
        vision_tower_aux = self.vision_backbone.get_vision_tower_aux()
        if vision_tower_aux is not None and images_aux is not None and bbox_list is not None and len(bbox_list) > 0:
            started = time.perf_counter()
            print("[VLM-FO1] Analysing the proposed regions...", file=sys.stderr, flush=True)
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
            print(f"[VLM-FO1] Region features ready ({time.perf_counter() - started:.1f}s).", file=sys.stderr, flush=True)

        return image_features, region_features

    def _prepare_qwen_inputs(
        self,
        input_ids: torch.LongTensor,
        images=None,
        images_aux=None,
        bbox_list=None,
        image_grid_thws=None,
        attention_mask=None,
    ):
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a non-empty [batch, tokens] tensor.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids.")
            attention_mask = attention_mask.bool()

        image_features, region_features = self._encode_visual_features(
            images,
            images_aux=images_aux,
            bbox_list=bbox_list,
            image_grid_thws=image_grid_thws,
        )

        prompt_embeds = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_input_ids = cur_input_ids[attention_mask[batch_idx].to(cur_input_ids.device)]
            if cur_input_ids.numel() == 0:
                raise ValueError("Each prompt must contain at least one unmasked token.")
            special_positions = torch.where(
                (cur_input_ids == IMAGE_TOKEN_INDEX) | (cur_input_ids == DEFAULT_REGION_INDEX)
            )[0].tolist()
            boundaries = [-1] + special_positions + [cur_input_ids.shape[0]]

            cur_embeds = []
            cur_region_idx = 0

            for idx in range(len(boundaries) - 1):
                text_ids = cur_input_ids[boundaries[idx] + 1:boundaries[idx + 1]]
                if text_ids.numel() > 0:
                    cur_embeds.append(self._embed_text(text_ids))

                if idx == len(boundaries) - 2:
                    continue

                special_token = cur_input_ids[boundaries[idx + 1]].item()
                if special_token == IMAGE_TOKEN_INDEX:
                    if cur_image_idx >= len(image_features):
                        raise ValueError("Prompt contains more <image> markers than prepared images.")
                    feature = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_embeds.append(feature)
                elif special_token == DEFAULT_REGION_INDEX:
                    if region_features is None:
                        raise ValueError("Prompt contains <regionfeat> markers but no region features were prepared.")
                    if batch_idx >= len(region_features) or cur_region_idx >= len(region_features[batch_idx]):
                        raise ValueError("Prompt contains more <regionfeat> markers than prepared regions.")
                    feature = region_features[batch_idx][cur_region_idx].unsqueeze(0)
                    cur_region_idx += 1
                    cur_embeds.append(feature)

            prompt_embeds.append(torch.cat(cur_embeds, dim=0))

        if cur_image_idx != len(image_features):
            raise ValueError("Prepared images must have matching <image> markers in the prompt.")

        max_len = max(embed.shape[0] for embed in prompt_embeds)
        batch_size = len(prompt_embeds)
        padded_embeds = []
        padded_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)

        # Left-pad: generation appends after the last position, so shorter
        # prompts must end flush with the batch's right edge.
        for idx, embed in enumerate(prompt_embeds):
            pad_len = max_len - embed.shape[0]
            padded_embeds.append(
                torch.cat(
                    [
                        torch.zeros(
                            (pad_len, embed.shape[1]),
                            dtype=embed.dtype,
                            device=embed.device,
                        ),
                        embed,
                    ],
                    dim=0,
                )
            )
            padded_attention_mask[idx, pad_len:] = 1

        return torch.stack(padded_embeds, dim=0), padded_attention_mask

    @torch.inference_mode()
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

        inputs_embeds, qwen_attention_mask = self._prepare_qwen_inputs(
            input_ids,
            images=images,
            images_aux=images_aux,
            bbox_list=bbox_list,
            image_grid_thws=image_grid_thws,
            attention_mask=attention_mask,
        )

        generation_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=qwen_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=use_cache,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            streamer=streamer,
        )
        if stopping_criteria:
            generation_kwargs["stopping_criteria"] = [
                _PromptAwareStoppingCriteria(criterion, input_ids) for criterion in stopping_criteria
            ]
        if do_sample:
            if temperature <= 0:
                raise ValueError("Sampling requires temperature > 0.")
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        generation_kwargs.update(kwargs)
        started = time.perf_counter()
        print(f"[VLM-FO1] Generating a response (up to {max_new_tokens} tokens)...", file=sys.stderr, flush=True)
        generated = self.language_model.generate(**generation_kwargs)
        print(f"[VLM-FO1] Response ready ({time.perf_counter() - started:.1f}s).", file=sys.stderr, flush=True)
        generated_ids = generated.sequences if hasattr(generated, "sequences") else generated

        # With inputs_embeds and no input_ids, generate() returns only the new
        # tokens. Callers slice the output by the prompt length
        # (output_ids[:, inputs.shape[1]:]), so prepend the original prompt
        # ids — not the longer internal visual sequence — to keep that
        # slicing correct.
        repeats = generated_ids.shape[0] // input_ids.shape[0]
        prompt_ids = input_ids.repeat_interleave(repeats, dim=0).to(generated_ids.device)
        sequences = torch.cat([prompt_ids, generated_ids], dim=1)
        if hasattr(generated, "sequences"):
            generated.sequences = sequences
            return generated
        return sequences
