"""Training: chat formatting, collation, LoRA/QLoRA setup and the training loop.

Heavy dependencies (torch, transformers, peft) are imported lazily inside the
functions that need them so that ``import rineya_lab`` works on a CPU-only box
with none of them installed.
"""

from rineya_lab.training.chat import PORTABLE_CHAT_TEMPLATE, ChatFormatter, EncodedExample

__all__ = ["ChatFormatter", "EncodedExample", "PORTABLE_CHAT_TEMPLATE"]
