"""Model entry points; import the large vision stack only when it is needed."""

__all__ = ["OmChatQwen25VLForCausalLM", "OmChatQwen25VLConfig"]


def __getattr__(name):
    if name in __all__:
        from .language_model import omchat_qwen2_5_vl

        return getattr(omchat_qwen2_5_vl, name)
    raise AttributeError(name)
