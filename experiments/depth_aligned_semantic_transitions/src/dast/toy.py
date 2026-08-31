"""Offline toy model and tokenizer for deterministic engineering qualification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class HashTokenizer:
    """A fixed-vocabulary tokenizer with a prefix-stable chat template."""

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    padding_side = "right"

    def __init__(self, vocab_size: int = 1024) -> None:
        if vocab_size < 32:
            raise ValueError("vocab_size must be at least 32")
        self.vocab_size = vocab_size

    def _pieces(self, text: str) -> list[str]:
        normalized = text.replace("\n", " \n ")
        return [piece for piece in normalized.split() if piece]

    def _id(self, piece: str) -> int:
        digest = hashlib.sha256(piece.encode("utf-8")).digest()
        return 3 + int.from_bytes(digest[:8], "big") % (self.vocab_size - 3)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = [self._id(piece) for piece in self._pieces(text)]
        return ([self.bos_token_id] + ids) if add_special_tokens else ids

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
        **_: Any,
    ) -> list[int] | str:
        del enable_thinking
        ids = [self.bos_token_id]
        text_parts: list[str] = []
        for message in messages:
            role = message["role"].upper()
            marker = f"<{role}>"
            text_parts.extend([marker, message["content"], "</TURN>"])
            ids.extend(self.encode(marker))
            ids.extend(self.encode(message["content"]))
            ids.extend(self.encode("</TURN>"))
        if add_generation_prompt:
            text_parts.append("<ASSISTANT>")
            ids.extend(self.encode("<ASSISTANT>"))
        else:
            ids.append(self.eos_token_id)
        return ids if tokenize else " ".join(text_parts)

    def __call__(
        self,
        texts: Sequence[str] | str,
        *,
        padding: bool = True,
        truncation: bool = True,
        max_length: int | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        if isinstance(texts, str):
            texts = [texts]
        rows = [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        width = max((len(row) for row in rows), default=0) if padding else 0
        ids = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": masks}

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        # Hash tokenization is intentionally irreversible.  This method exists so
        # generic generation code can be exercised; it is not a quality scorer.
        values = [int(item) for item in ids]
        if skip_special_tokens:
            values = [item for item in values if item > self.bos_token_id]
        return " ".join(f"tok{item}" for item in values)


class ToyBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.context = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=hidden.dtype)
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        cumulative = torch.cumsum(hidden * mask, dim=1)
        counts = torch.cumsum(mask, dim=1).clamp_min(1.0)
        context = cumulative / counts
        hidden = self.norm1(hidden + self.context(context))
        return self.norm2(hidden + self.ffn(hidden))


@dataclass
class ToyOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    hidden_states: tuple[torch.Tensor, ...] | None


class ToyCausalLM(nn.Module):
    """Small causal residual network exposing the Hugging Face subset DAST uses."""

    def __init__(self, vocab_size: int = 1024, hidden_size: int = 48, num_hidden_layers: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            use_cache=False,
        )
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(ToyBlock(hidden_size) for _ in range(num_hidden_layers))
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        use_cache: bool = False,
        **_: Any,
    ) -> ToyOutput:
        del return_dict, use_cache
        hidden = self.embed_tokens(input_ids)
        all_hidden: list[torch.Tensor] | None = [hidden] if output_hidden_states else None
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=attention_mask)
            if all_hidden is not None:
                all_hidden.append(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        return ToyOutput(logits=logits, loss=loss, hidden_states=tuple(all_hidden) if all_hidden is not None else None)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        max_new_tokens: int = 16,
        **_: Any,
    ) -> torch.Tensor:
        result = input_ids
        mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        for _ in range(max_new_tokens):
            output = self(
                input_ids=result,
                attention_mask=mask,
                output_hidden_states=False,
                return_dict=True,
            )
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            result = torch.cat([result, next_token], dim=1)
            mask = torch.cat([mask, torch.ones_like(next_token)], dim=1)
        return result
