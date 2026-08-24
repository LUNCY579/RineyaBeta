import json
from pathlib import Path

import pytest
import torch

from pdrr.data import collate, deterministic_order, load_chat_jsonl


class Tokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "|".join(message["content"] for message in messages) + "|assistant:"

    def __call__(self, text, add_special_tokens=False):
        return type("Tokenized", (), {"input_ids": [1 + (ord(char) % 13) for char in text]})


def write_rows(path: Path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_deterministic_order_changes_by_epoch_but_repeats_exactly():
    first = deterministic_order(20, seed=17, epoch=0)
    assert first == deterministic_order(20, seed=17, epoch=0)
    assert first != deterministic_order(20, seed=17, epoch=1)


def test_chat_loader_and_collate(tmp_path):
    path = tmp_path / "data.jsonl"
    write_rows(path, [{
        "id": "a", "category": "logic",
        "messages": [{"role": "user", "content": "x"},
                     {"role": "assistant", "content": "y"}],
    }])
    examples = load_chat_jsonl(path, Tokenizer(), max_length=32)
    batch = collate(examples, 0)
    assert batch["input_ids"].shape[0] == 1
    assert torch.any(batch["labels"] != -100)


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "data.jsonl"
    row = {"id": "same", "messages": [{"role": "user", "content": "x"},
                                         {"role": "assistant", "content": "y"}]}
    write_rows(path, [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        load_chat_jsonl(path, Tokenizer(), max_length=32)
