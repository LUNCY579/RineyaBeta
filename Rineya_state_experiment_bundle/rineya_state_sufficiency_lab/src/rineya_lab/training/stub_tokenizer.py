"""A dependency-free byte tokenizer used for offline smoke tests.

Selected with ``model.name_or_path: "builtin:byte"``. It implements exactly the
surface the lab uses (``__call__``, special-token ids, ``padding_side``) and
deliberately exposes no ``chat_template`` so the portable fallback path is
exercised. It is a test fixture, never a research configuration.
"""

from __future__ import annotations

from typing import Any

BUILTIN_TOKENIZER_ID = "builtin:byte"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
OFFSET = 4
VOCAB_SIZE = OFFSET + 256


class ByteTokenizer:
    name_or_path = BUILTIN_TOKENIZER_ID
    chat_template = None
    padding_side = "right"

    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    pad_token_id = PAD_ID
    bos_token_id = BOS_ID
    eos_token_id = EOS_ID
    unk_token_id = UNK_ID
    vocab_size = VOCAB_SIZE
    model_max_length = 1 << 20

    def __call__(self, text: str | list[str], add_special_tokens: bool = True, **_: Any) -> dict[str, Any]:
        if isinstance(text, list):
            return {"input_ids": [self._encode(item, add_special_tokens) for item in text]}
        return {"input_ids": self._encode(text, add_special_tokens)}

    def _encode(self, text: str, add_special_tokens: bool) -> list[int]:
        ids = [OFFSET + byte for byte in text.encode("utf-8")]
        if add_special_tokens:
            ids = [BOS_ID, *ids, EOS_ID]
        return ids

    def encode(self, text: str, add_special_tokens: bool = True, **_: Any) -> list[int]:
        return self._encode(text, add_special_tokens)

    def decode(self, ids: list[int], skip_special_tokens: bool = True, **_: Any) -> str:
        payload = bytearray()
        for token_id in ids:
            if token_id < OFFSET:
                if not skip_special_tokens and token_id == EOS_ID:
                    break
                continue
            payload.append(token_id - OFFSET)
        return payload.decode("utf-8", errors="replace")

    def batch_decode(self, sequences, skip_special_tokens: bool = True, **kwargs: Any) -> list[str]:
        return [self.decode(list(seq), skip_special_tokens=skip_special_tokens, **kwargs) for seq in sequences]

    def __len__(self) -> int:
        return VOCAB_SIZE

    def save_pretrained(self, directory: str) -> None:
        from pathlib import Path

        from rineya_lab.utils.io import write_json

        write_json(Path(directory) / "tokenizer_stub.json", {"kind": BUILTIN_TOKENIZER_ID, "vocab_size": VOCAB_SIZE})


def is_builtin(name_or_path: str) -> bool:
    return name_or_path == BUILTIN_TOKENIZER_ID
