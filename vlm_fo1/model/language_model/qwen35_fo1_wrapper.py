import warnings

import torch
import torch.nn as nn

from vlm_fo1.constants import DEFAULT_REGION_INDEX, IMAGE_TOKEN_INDEX


class Qwen35FO1Wrapper(nn.Module):
    def __init__(self, vision_backbone, language_model, tokenizer):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.language_model = language_model
        self.tokenizer = tokenizer
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

    def _embed_text(self, token_ids):
        token_ids = token_ids.to(self.device)
        return self.language_model.get_input_embeddings()(token_ids)

    def prepare_embeddings(
        self,
        input_ids,
        images=None,
        images_aux=None,
        bbox_list=None,
        image_grid_thws=None,
        attention_mask=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        image_features, region_features = self.encode_visual_features(
            images,
            images_aux=images_aux,
            bbox_list=bbox_list,
            image_grid_thws=image_grid_thws,
        )

        batch_embeddings = []
        image_index = 0

        for batch_index, tokens in enumerate(input_ids):
            tokens = tokens[attention_mask[batch_index].to(tokens.device)]
            marker_positions = torch.where(
                (tokens == IMAGE_TOKEN_INDEX) | (tokens == DEFAULT_REGION_INDEX)
            )[0]

            pieces = []
            text_start = 0
            region_index = 0

            for marker_position in marker_positions.tolist():
                text = tokens[text_start:marker_position]
                if text.numel() > 0:
                    pieces.append(self._embed_text(text))

                marker = tokens[marker_position].item()
                if marker == IMAGE_TOKEN_INDEX:
                    if image_index >= len(image_features):
                        raise ValueError("There are more image markers than images")
                    pieces.append(image_features[image_index])
                    image_index += 1
                else:
                    if region_features is None:
                        raise ValueError("Region markers were used without region features")
                    pieces.append(
                        region_features[batch_index][region_index].unsqueeze(0)
                    )
                    region_index += 1

                text_start = marker_position + 1

            remaining_text = tokens[text_start:]
            if remaining_text.numel() > 0:
                pieces.append(self._embed_text(remaining_text))

            batch_embeddings.append(torch.cat(pieces, dim=0))

        max_length = max(item.shape[0] for item in batch_embeddings)
        padded_embeddings = []
        new_attention_mask = torch.zeros(
            len(batch_embeddings),
            max_length,
            dtype=torch.long,
            device=self.device,
        )

        for batch_index, embeddings in enumerate(batch_embeddings):
            length = embeddings.shape[0]
            padding = torch.zeros(
                max_length - length,
                embeddings.shape[1],
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            padded_embeddings.append(torch.cat((embeddings, padding), dim=0))
            new_attention_mask[batch_index, :length] = 1

        return torch.stack(padded_embeddings), new_attention_mask

    def generate(
        self,
        inputs=None,
        input_ids=None,
        images=None,
        images_aux=None,
        bbox_list=None,
        image_grid_thws=None,
        attention_mask=None,
        max_new_tokens=512,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
        streamer=None,
        stopping_criteria=None,
        **kwargs,
    ):
        input_ids = input_ids if input_ids is not None else inputs
        if input_ids is None:
            raise ValueError("generate() needs input_ids")

        embeddings, new_attention_mask = self.prepare_embeddings(
            input_ids,
            images=images,
            images_aux=images_aux,
            bbox_list=bbox_list,
            image_grid_thws=image_grid_thws,
            attention_mask=attention_mask,
        )

        if stopping_criteria is not None:
            warnings.warn(
                "The old stopping criteria cannot be used after expanding the "
                "prompt embeddings. Generation will stop at the EOS token.",
                RuntimeWarning,
            )

        generation_args = {
            "inputs_embeds": embeddings,
            "attention_mask": new_attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "top_p": top_p,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "streamer": streamer,
        }
        if do_sample:
            generation_args["temperature"] = temperature
        generation_args.update(kwargs)

        generated_ids = self.language_model.generate(**generation_args)

        first_token = generated_ids[:, :1]
        seed_tokens = [self.tokenizer.pad_token_id]
        if self.tokenizer.bos_token_id is not None:
            seed_tokens.append(self.tokenizer.bos_token_id)
        seed_tokens = torch.tensor(seed_tokens, device=generated_ids.device)
        if torch.isin(first_token, seed_tokens).all():
            generated_ids = generated_ids[:, 1:]

        return torch.cat((input_ids.to(generated_ids.device), generated_ids), dim=1)
