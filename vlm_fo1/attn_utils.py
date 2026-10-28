import os

_ENV_VAR = "VLM_FO1_ATTN_IMPL"


def get_attn_implementation(device=None) -> str:
    """Attention backend for the language model and the Qwen2.5-VL vision tower.

    SDPA is the portable default, including CPU and Colab T4. FlashAttention
    requires a compatible GPU and is available through VLM_FO1_ATTN_IMPL.
    """
    impl = os.environ.get(_ENV_VAR)
    if impl:
        if impl not in {"flash_attention_2", "sdpa", "eager"}:
            raise ValueError(f"{_ENV_VAR} must be flash_attention_2, sdpa or eager.")
        if impl == "flash_attention_2" and device is not None and not str(device).startswith("cuda"):
            raise ValueError("flash_attention_2 requires a CUDA device; use sdpa for CPU inference.")
        return impl
    return "sdpa"
