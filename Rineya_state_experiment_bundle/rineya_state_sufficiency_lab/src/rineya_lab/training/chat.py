"""Chat-template handling with a portable fallback, and assistant-only encoding.

Two properties matter for this experiment and are enforced here.

**Assistant-only loss.** Only the final assistant turn contributes to the loss.
The prompt is masked with ``-100``. Because every dataset record is exactly one
user turn plus one assistant turn, the mask is computed exactly (prompt encoded
separately, then concatenated) rather than by searching for template markers,
which is fragile across tokenizers.

**Target-preserving truncation.** If a record exceeds ``max_seq_len`` the prompt
is shortened, never the supervised target. Silently truncating targets would
change what SPLIT and FULL are actually supervised on -- FULL targets are ~4x
longer, so a naive right-truncation would systematically damage exactly one arm
of the comparison. Truncation events are counted and reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rineya_lab.utils.logging import get_logger

LOGGER = get_logger("training.chat")

IGNORE_INDEX = -100

# A deliberately simple, tokenizer-agnostic template used when the base model
# ships no chat template. It is plain text (no Jinja) so it renders identically
# in training and evaluation regardless of the transformers version.
PORTABLE_CHAT_TEMPLATE = {
    "system_prefix": "### System:\n",
    "user_prefix": "### User:\n",
    "assistant_prefix": "### Assistant:\n",
    "turn_suffix": "\n\n",
}

MIN_PROMPT_TOKENS = 8


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    prompt_token_count: int
    target_token_count: int
    prompt_truncated: bool = False
    target_truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.input_ids)


class ChatFormatter:
    """Renders message lists to text and encodes them with an assistant-only mask."""

    def __init__(self, tokenizer, model_cfg=None) -> None:
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg
        self.template_source = "none"
        self.template_kwargs: dict[str, Any] = dict(getattr(model_cfg, "chat_template_kwargs", {}) or {})
        self._configure_template()

    # -- template selection --------------------------------------------------------

    def _configure_template(self) -> None:
        override = getattr(self.model_cfg, "chat_template_override", None) if self.model_cfg else None
        if override:
            self.tokenizer.chat_template = override
            self.template_source = "config_override"
            return
        if getattr(self.tokenizer, "chat_template", None):
            self.template_source = "tokenizer_chat_template"
            return
        allow_fallback = getattr(self.model_cfg, "use_portable_template_fallback", True) if self.model_cfg else True
        if not allow_fallback:
            raise ValueError(
                "The tokenizer has no chat_template and model.use_portable_template_fallback is false. "
                "Set model.chat_template_override or enable the portable fallback."
            )
        self.template_source = "portable_fallback"
        LOGGER.warning(
            "Tokenizer %s has no chat template; using the portable fallback format. "
            "Both arms of the comparison use the same fallback, so the contrast stays valid.",
            getattr(self.tokenizer, "name_or_path", "<unknown>"),
        )

    def template_info(self) -> dict[str, Any]:
        return {
            "template_source": self.template_source,
            "template_kwargs": self.template_kwargs,
            "portable_fallback": PORTABLE_CHAT_TEMPLATE if self.template_source == "portable_fallback" else None,
            "eos_token": getattr(self.tokenizer, "eos_token", None),
            "pad_token": getattr(self.tokenizer, "pad_token", None),
            "bos_token": getattr(self.tokenizer, "bos_token", None),
        }

    # -- rendering -----------------------------------------------------------------

    def _render_portable(self, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
        parts: list[str] = []
        for message in messages:
            role = message.get("role", "user")
            prefix = {
                "system": PORTABLE_CHAT_TEMPLATE["system_prefix"],
                "user": PORTABLE_CHAT_TEMPLATE["user_prefix"],
                "assistant": PORTABLE_CHAT_TEMPLATE["assistant_prefix"],
            }.get(role, PORTABLE_CHAT_TEMPLATE["user_prefix"])
            parts.append(f"{prefix}{message['content']}{PORTABLE_CHAT_TEMPLATE['turn_suffix']}")
        if add_generation_prompt:
            parts.append(PORTABLE_CHAT_TEMPLATE["assistant_prefix"])
        return "".join(parts)

    def render_prompt(self, messages: list[dict[str, str]]) -> str:
        """Render everything up to (and including) the generation prompt."""
        if self.template_source == "portable_fallback":
            return self._render_portable(messages, add_generation_prompt=True)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.template_kwargs
        )

    def render_full(self, messages: list[dict[str, str]]) -> str:
        if self.template_source == "portable_fallback":
            return self._render_portable(messages, add_generation_prompt=False)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, **self.template_kwargs
        )

    # -- encoding ------------------------------------------------------------------

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def encode_prompt(self, messages: list[dict[str, str]]) -> list[int]:
        ids = self._encode(self.render_prompt(messages))
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if bos is not None and (not ids or ids[0] != bos) and self.template_source == "portable_fallback":
            ids = [bos, *ids]
        return ids

    def encode_example(
        self,
        messages: list[dict[str, str]],
        *,
        max_seq_len: int,
        prompt_head_keep: int = 64,
    ) -> EncodedExample:
        """Encode a message list into ``input_ids`` and assistant-only ``labels``."""
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise ValueError("encode_example expects a message list ending with an assistant turn")

        prompt_ids = self.encode_prompt(messages[:-1])
        target_text = messages[-1]["content"]
        target_ids = self._encode(target_text)

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
            target_ids = [*target_ids, eos_id]

        prompt_truncated = False
        target_truncated = False

        if len(prompt_ids) + len(target_ids) > max_seq_len:
            room_for_prompt = max_seq_len - len(target_ids)
            if room_for_prompt >= MIN_PROMPT_TOKENS:
                head = min(prompt_head_keep, room_for_prompt)
                tail = room_for_prompt - head
                # Keep the head (role markers / problem opening) and the tail
                # (the compact state and the instruction, which carry the signal).
                prompt_ids = prompt_ids[:head] + (prompt_ids[-tail:] if tail > 0 else [])
                prompt_truncated = True
            else:
                # The target alone does not fit. Keep a minimal prompt and cut the
                # target from the right; this is counted and surfaced in the report.
                prompt_ids = prompt_ids[:MIN_PROMPT_TOKENS]
                keep = max_seq_len - len(prompt_ids)
                target_ids = target_ids[:keep]
                if eos_id is not None and target_ids and target_ids[-1] != eos_id:
                    target_ids[-1] = eos_id
                prompt_truncated = True
                target_truncated = True

        input_ids = [*prompt_ids, *target_ids]
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(target_ids)
        return EncodedExample(
            input_ids=input_ids,
            labels=labels,
            prompt_token_count=len(prompt_ids),
            target_token_count=len(target_ids),
            prompt_truncated=prompt_truncated,
            target_truncated=target_truncated,
        )


def ensure_pad_token(tokenizer) -> dict[str, Any]:
    """Guarantee a pad token without resizing embeddings.

    Reusing EOS as PAD is safe here because padded positions are masked out of
    both the attention mask and the labels.
    """
    info = {"pad_token_added": False, "pad_equals_eos": False}
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
            info["pad_token_added"] = True
            info["pad_equals_eos"] = True
        else:  # pragma: no cover - exotic tokenizers
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            info["pad_token_added"] = True
    else:
        info["pad_equals_eos"] = tokenizer.pad_token_id == getattr(tokenizer, "eos_token_id", None)
    tokenizer.padding_side = "right"
    return info
