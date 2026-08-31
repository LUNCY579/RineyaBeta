"""Shared semantic interface and frozen text-derived target geometry."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class SharedStateProjector(nn.Module):
    """One projector shared by every checkpoint: the semantic ABI across depth."""

    def __init__(self, hidden_size: int, state_dim: int, projector_hidden_dim: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projector_hidden_dim),
            nn.GELU(),
            nn.Linear(projector_hidden_dim, state_dim),
        )

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        # Auxiliary heads remain fp32 even when the frozen/LoRA backbone is bf16.
        return F.normalize(self.network(pooled_hidden.float()), dim=-1)


class FrozenTextTargetEncoder:
    """Map target text into a deterministic, non-trainable semantic coordinate system.

    V1 intentionally grounds targets in the frozen base input-embedding geometry
    rather than claiming a discovered symbolic language.  A fixed random
    projection removes dependence on the model hidden width and is regenerated
    exactly from ``projection_seed``.
    """

    def __init__(
        self,
        embedding: nn.Module,
        tokenizer: Any,
        *,
        state_dim: int,
        projection_seed: int,
        max_tokens: int = 64,
        allow_trainable_embedding: bool = False,
    ) -> None:
        if not hasattr(embedding, "weight"):
            raise TypeError("input embedding must expose a weight tensor")
        weight = embedding.weight
        if weight.ndim != 2:
            raise ValueError("embedding weight must be rank 2")
        if weight.requires_grad and not allow_trainable_embedding:
            raise ValueError(
                "semantic target embedding is trainable; freeze it or explicitly opt into target drift"
            )
        self.embedding = embedding
        self.tokenizer = tokenizer
        self.state_dim = state_dim
        self.max_tokens = max_tokens
        generator = torch.Generator(device="cpu")
        generator.manual_seed(projection_seed)
        projection = torch.randn(weight.shape[1], state_dim, generator=generator, dtype=torch.float32)
        if weight.shape[1] >= state_dim:
            projection, _ = torch.linalg.qr(projection, mode="reduced")
        else:
            projection = F.normalize(projection, dim=0)
        self.projection = projection.contiguous()

    def _tokenize(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
            add_special_tokens=False,
        )
        if isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
        else:
            input_ids = encoded.input_ids
            attention_mask = getattr(encoded, "attention_mask", None)
        if attention_mask is None:
            pad = getattr(self.tokenizer, "pad_token_id", None)
            attention_mask = torch.ones_like(input_ids) if pad is None else input_ids.ne(int(pad)).long()
        return input_ids.long(), attention_mask.long()

    @torch.no_grad()
    def encode(self, texts: Sequence[str], *, device: torch.device | str) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.state_dim), dtype=torch.float32, device=device)
        input_ids, attention_mask = self._tokenize(texts)
        embedding_device = self.embedding.weight.device
        input_ids = input_ids.to(embedding_device)
        attention_mask = attention_mask.to(embedding_device)
        hidden = self.embedding(input_ids).float()
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1.0)
        pooled = (hidden * mask).sum(dim=1) / count
        projected = pooled @ self.projection.to(pooled.device)
        return F.normalize(projected, dim=-1).to(device=device)

    @torch.no_grad()
    def encode_grid(self, texts: Sequence[Sequence[str]], *, device: torch.device | str) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 0, self.state_dim), dtype=torch.float32, device=device)
        width = len(texts[0])
        if any(len(row) != width for row in texts):
            raise ValueError("semantic target grid must be rectangular")
        flattened = [item for row in texts for item in row]
        encoded = self.encode(flattened, device=device)
        return encoded.reshape(len(texts), width, self.state_dim)
