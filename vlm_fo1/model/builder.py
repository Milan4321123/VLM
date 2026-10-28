import warnings

import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch
from vlm_fo1.model.language_model.qwen35_fo1_wrapper import Qwen35FO1InferenceWrapper
import os
from vlm_fo1.attn_utils import get_attn_implementation


def _get_qwen35_auto_model_class():
    """Load the official text decoder; FO1 supplies all visual embeddings.

    The Transformers text-only class extracts text_config and remaps the
    published multimodal checkpoint's language weights. Its unused native
    vision tower is never allocated (about 384 MiB saved on float32 CPU).
    Keep this helper's existing name for callers of the previous loader.
    """
    model_class = getattr(transformers, "Qwen3_5ForCausalLM", None)
    if model_class is not None:
        return model_class
    raise ImportError(
        "Could not find Qwen3_5ForCausalLM. Install the supported Transformers "
        "version from requirements.txt to load Qwen/Qwen3.5-0.8B."
    )


def _preferred_torch_dtype(device):
    """Choose a dtype supported by the active device (T4 uses float16)."""
    if str(device) == "cpu":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _qwen35_device_map(device):
    if device is None or device == "auto":
        return "auto"
    if isinstance(device, str) and device.startswith("cuda"):
        return {"": device}
    if isinstance(device, str) and device in {"cpu", "mps"}:
        return {"": device}
    return "auto"


def _ensure_qwen35_fo1_tokens(tokenizer, num_region_tokens=100):
    """
    The FO1 checkpoint's tokenizer ships with <region0>..<region99> already in
    the vocabulary; Qwen3.5's does not, so add them before the embedding table
    is resized. The new rows are untrained, but tokenization has to be stable
    for region-index prompts and outputs either way.
    """
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})

    region_tokens = [f"<region{i}>" for i in range(num_region_tokens)]
    return tokenizer.add_tokens(region_tokens, special_tokens=True)


def _load_qwen35_language_model(llm_model_path, tokenizer, load_8bit=False, load_4bit=False, device="cuda"):
    model_class = _get_qwen35_auto_model_class()
    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": _qwen35_device_map(device),
        "attn_implementation": get_attn_implementation(device),
        "output_loading_info": True,
    }

    if load_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        kwargs["torch_dtype"] = _preferred_torch_dtype(device)

    language_model, loading_info = model_class.from_pretrained(llm_model_path, **kwargs)
    if loading_info.get("missing_keys") or loading_info.get("mismatched_keys") or loading_info.get("error_msgs"):
        raise RuntimeError(
            "Incomplete Qwen3.5 checkpoint: refusing to run with randomly initialized "
            f"decoder components. Loading details: {loading_info}"
        )
    # Qwen checkpoints pad the embedding table past the tokenizer length, so
    # only ever grow it — shrinking would throw away pretrained rows.
    if len(tokenizer) > language_model.get_input_embeddings().num_embeddings:
        language_model.resize_token_embeddings(len(tokenizer))
    return language_model


def load_pretrained_model(
    model_path,
    load_8bit=False,
    load_4bit=False,
    device="cuda",
    llm_model_path=None,
    bridge_state_path=None,
    bridge_seed=0,
):
    """
    Loads a pretrained model along with its vision towers (and associated image processors).
    This function supports loading in 8bit/4bit precision and explicit device placement.

    Args:
        model_path (str): Path to the pretrained model directory.
        load_8bit (bool): Whether to load the model in 8bit mode.
        load_4bit (bool): Whether to load the model in 4bit mode.
        device (str): Device to load model onto, e.g., "cuda" or "cpu".
        llm_model_path (str): Optional replacement decoder (e.g. Qwen3.5).
            When set, the checkpoint is used purely as a vision stack and all
            generation runs through this model. Default keeps the
            checkpoint's own decoder.
        bridge_state_path (str): Optional trained state for the feature
            bridge used with a replacement decoder.

    Returns:
        tuple: (tokenizer, model, image_processor)
    """
    from vlm_fo1.model import OmChatQwen25VLForCausalLM, OmChatQwen25VLConfig

    device = str(device)
    if load_8bit and load_4bit:
        raise ValueError("Choose either 4-bit or 8-bit quantization, not both.")
    if (load_8bit or load_4bit) and not device.startswith("cuda"):
        raise ValueError("The supported bitsandbytes quantized path requires CUDA.")
    model_path = os.fspath(model_path)
    model_dtype = _preferred_torch_dtype(device)
    kwargs = {"device_map": device}

    # Set model loading parameters for quantization or floating point
    if load_8bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        kwargs['torch_dtype'] = model_dtype

    config = OmChatQwen25VLConfig.from_pretrained(model_path)
    if not getattr(config, "mm_vision_tower", None):
        raise ValueError("The checkpoint must contain the VLM-FO1 visual stack (mm_vision_tower).")

    # The tokenizer has to match the model that generates: the chat marker
    # ids, eos/pad handling and vocabulary size all come from it. So when a
    # replacement LLM is configured, load the tokenizer from there instead of
    # from the FO1 checkpoint.
    tokenizer_source = llm_model_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False, trust_remote_code=True)
    if llm_model_path:
        added_tokens = _ensure_qwen35_fo1_tokens(
            tokenizer,
            num_region_tokens=getattr(config, "mm_num_region_tokens", 100),
        )
        if added_tokens:
            warnings.warn(
                f"Added {added_tokens} VLM-FO1 region tokens to the Qwen3.5 tokenizer. "
                "The added token meanings have no trained alignment with Qwen3.5.",
                RuntimeWarning,
            )

    # If this is the Qwen2.5-VL variant, load with additional kwargs
    model, loading_info = OmChatQwen25VLForCausalLM.from_pretrained(
        model_path,
        config=config,
        low_cpu_mem_usage=True,
        output_loading_info=True,
        vision_only=bool(llm_model_path),
        attn_implementation=get_attn_implementation(device),
        **kwargs
    )
    if loading_info.get("missing_keys") or loading_info.get("mismatched_keys"):
        raise RuntimeError(
            "Incomplete VLM-FO1 checkpoint: refusing to run with randomly initialized "
            f"pretrained components. Loading details: {loading_info}"
        )

    # --- Vision Tower Loading ---
    # Load the main vision tower weights from model_path if it is not yet loaded
    primary_image_processor = None
    primary_vision_tower = model.get_vision_tower()
    if primary_vision_tower and not primary_vision_tower.is_loaded:
        primary_vision_tower.load_model(model_path=model_path, is_train=False)
        primary_vision_tower.to(device=device, dtype=model_dtype)  # Move to correct device/dtype

    # Grab primary image processor from vision tower, if present
    if primary_vision_tower:
        primary_image_processor = primary_vision_tower.image_processor

    # --- Auxiliary Vision Tower Handling (Qwen2.5-VL case only) ---
    aux_vision_tower = None
    aux_image_processor = None
    if model.get_vision_tower_aux() is not None:
        try:
            aux_image_size = model.config.aux_image_size
        except Exception:
            # If aux_image_size is missing from config fallback to 768
            aux_image_size = 768

        aux_image_aspect_ratio = model.config.aux_image_aspect_ratio
        aux_vision_tower = model.get_vision_tower_aux()
        # Only load if not already loaded
        if aux_vision_tower and not aux_vision_tower.is_loaded:
            aux_vision_tower.load_model(image_size=aux_image_size, is_train=False, aspect_ratio=aux_image_aspect_ratio)
            aux_vision_tower.to(device=device, dtype=model_dtype)

    # Get auxiliary image processor if there is an aux vision tower
    if aux_vision_tower:
        aux_image_processor = aux_vision_tower.image_processor

    # image_processor returned as a tuple of (primary, aux)
    image_processor = (primary_image_processor, aux_image_processor)

    if llm_model_path:
        # Keep the checkpoint purely as a vision stack and route generation
        # through the replacement LLM.
        # The original decoder was omitted before loading to limit peak memory.
        language_model = _load_qwen35_language_model(
            llm_model_path,
            tokenizer,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            device=device,
        )
        bridge_state_path = bridge_state_path or os.environ.get("VLM_FO1_QWEN35_BRIDGE_PATH")
        model = Qwen35FO1InferenceWrapper(
            vision_backbone=model,
            language_model=language_model,
            tokenizer=tokenizer,
            bridge_state_path=bridge_state_path,
            bridge_seed=bridge_seed,
        )
        model.eval()
        return tokenizer, model, image_processor

    # Set original Qwen2.5-VL model to eval mode and move to correct device before returning
    model.eval()
    model.requires_grad_(False)
    if not (load_8bit or load_4bit):
        model.to(device=device, dtype=model_dtype)
    return tokenizer, model, image_processor
