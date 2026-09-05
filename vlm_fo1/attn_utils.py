import importlib.util
import os
import warnings

_ENV_VAR = "VLM_FO1_ATTN_IMPL"


def get_attn_implementation() -> str:
    """Attention backend for the language model and the Qwen2.5-VL vision tower.

    Order of precedence:
      1. the VLM_FO1_ATTN_IMPL environment variable ("flash_attention_2", "sdpa" or "eager"),
      2. "flash_attention_2" when the flash_attn package is installed (the original default),
      3. "sdpa" otherwise, with a warning.
    """
    impl = os.environ.get(_ENV_VAR)
    if impl:
        return impl
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    warnings.warn(
        "flash_attn is not installed; loading VLM-FO1 with attn_implementation='sdpa'. "
        f"Install flash-attn for the default backend or set {_ENV_VAR} to override."
    )
    return "sdpa"
