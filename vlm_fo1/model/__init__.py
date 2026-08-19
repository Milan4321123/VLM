# The two backbones need different transformers generations: the vendored
# Qwen2.5-VL code was copied from transformers 4.50.x, while the Qwen3.5
# classes import from transformers >= 5.2. Import whichever the installed
# transformers can support instead of failing the whole package.

_import_errors = {}

try:
    from .language_model.omchat_qwen2_5_vl import OmChatQwen25VLForCausalLM, OmChatQwen25VLConfig
except Exception as e:  # noqa: F841
    _import_errors["qwen2_5_vl"] = e

try:
    from .language_model.omchat_qwen3_5 import OmChatQwen35ForCausalLM, OmChatQwen35Config
except Exception as e:  # noqa: F841
    _import_errors["qwen3_5"] = e

if len(_import_errors) == 2:
    raise ImportError(
        "Neither the Qwen2.5-VL nor the Qwen3.5 backbone could be imported. "
        f"qwen2_5_vl: {_import_errors['qwen2_5_vl']!r} | qwen3_5: {_import_errors['qwen3_5']!r}"
    )
