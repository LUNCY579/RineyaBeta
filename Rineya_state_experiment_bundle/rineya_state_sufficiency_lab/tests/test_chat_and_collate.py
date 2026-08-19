from __future__ import annotations

import pytest

from rineya_lab.config import ModelConfig
from rineya_lab.training.chat import IGNORE_INDEX, ChatFormatter, ensure_pad_token
from rineya_lab.training.collate import pad_batch
from rineya_lab.training.schedule import lr_multiplier, resolve_warmup_steps
from rineya_lab.training.stub_tokenizer import ByteTokenizer

MESSAGES = [
    {"role": "user", "content": "Original problem:\nSolve 7x + 5 = 47.\n\nCurrent compact state:\n7x = 42."},
    {"role": "assistant", "content": "Divide both sides by 7: x = 6."},
]


@pytest.fixture
def formatter():
    tokenizer = ByteTokenizer()
    ensure_pad_token(tokenizer)
    return ChatFormatter(tokenizer, ModelConfig(name_or_path="builtin:byte"))


def test_portable_fallback_used_when_no_template(formatter):
    assert formatter.template_source == "portable_fallback"
    rendered = formatter.render_prompt(MESSAGES[:1])
    assert rendered.endswith("### Assistant:\n")
    assert "### User:" in rendered


def test_missing_template_can_be_made_fatal():
    tokenizer = ByteTokenizer()
    with pytest.raises(ValueError, match="portable"):
        ChatFormatter(tokenizer, ModelConfig(name_or_path="x", use_portable_template_fallback=False))


def test_assistant_only_masking(formatter):
    encoded = formatter.encode_example(MESSAGES, max_seq_len=1024)
    assert len(encoded.input_ids) == len(encoded.labels)
    assert all(label == IGNORE_INDEX for label in encoded.labels[: encoded.prompt_token_count])
    assert all(label != IGNORE_INDEX for label in encoded.labels[encoded.prompt_token_count :])
    supervised = encoded.labels[encoded.prompt_token_count :]
    assert supervised == encoded.input_ids[encoded.prompt_token_count :]


def test_target_ends_with_eos(formatter):
    encoded = formatter.encode_example(MESSAGES, max_seq_len=1024)
    assert encoded.input_ids[-1] == formatter.tokenizer.eos_token_id


def test_truncation_preserves_the_target(formatter):
    long_messages = [
        {"role": "user", "content": "x" * 4000},
        {"role": "assistant", "content": "The answer is 6."},
    ]
    encoded = formatter.encode_example(long_messages, max_seq_len=128, prompt_head_keep=16)
    assert len(encoded.input_ids) <= 128
    assert encoded.prompt_truncated and not encoded.target_truncated
    decoded = formatter.tokenizer.decode(encoded.input_ids[encoded.prompt_token_count :])
    assert "The answer is 6." in decoded


def test_truncation_keeps_prompt_head_and_tail(formatter):
    messages = [
        {"role": "user", "content": "HEAD-MARKER " + ("filler " * 2000) + " TAIL-MARKER"},
        {"role": "assistant", "content": "ok"},
    ]
    encoded = formatter.encode_example(messages, max_seq_len=200, prompt_head_keep=64)
    prompt_text = formatter.tokenizer.decode(encoded.input_ids[: encoded.prompt_token_count])
    assert "HEAD" in prompt_text and "TAIL-MARKER" in prompt_text


def test_oversized_target_is_flagged_not_silently_dropped(formatter):
    messages = [{"role": "user", "content": "short"}, {"role": "assistant", "content": "y" * 2000}]
    encoded = formatter.encode_example(messages, max_seq_len=64)
    assert encoded.target_truncated and len(encoded.input_ids) == 64


def test_encode_requires_assistant_last(formatter):
    with pytest.raises(ValueError):
        formatter.encode_example([{"role": "user", "content": "hi"}], max_seq_len=64)


def test_pad_batch_masks_padding(formatter):
    a = formatter.encode_example(MESSAGES, max_seq_len=512)
    b = formatter.encode_example(
        [{"role": "user", "content": "short"}, {"role": "assistant", "content": "ok"}], max_seq_len=512
    )
    batch = pad_batch([a, b], pad_token_id=formatter.tokenizer.pad_token_id, weights=[1.0, 0.5])
    assert len({len(row) for row in batch.input_ids}) == 1
    for mask, labels in zip(batch.attention_mask, batch.labels):
        for index, flag in enumerate(mask):
            if flag == 0:
                assert labels[index] == IGNORE_INDEX
    assert batch.padded_tokens > 0
    assert batch.weights == [1.0, 0.5]


def test_pad_batch_rounds_to_multiple(formatter):
    a = formatter.encode_example(MESSAGES, max_seq_len=512)
    batch = pad_batch([a], pad_token_id=0, pad_to_multiple_of=8)
    assert batch.meta["max_length"] % 8 == 0


def test_pad_batch_rejects_empty():
    with pytest.raises(ValueError):
        pad_batch([], pad_token_id=0)


@pytest.mark.parametrize("kind", ["constant", "linear", "cosine"])
def test_lr_schedule_shapes(kind):
    total, warmup = 100, 10
    assert lr_multiplier(0, total, warmup, kind) == pytest.approx(0.1)
    assert lr_multiplier(9, total, warmup, kind) == pytest.approx(1.0)
    end = lr_multiplier(total - 1, total, warmup, kind)
    assert 0.0 <= end <= 1.0
    if kind != "constant":
        assert end < 0.1


def test_warmup_resolution():
    assert resolve_warmup_steps(1000, 0.03, None) == 30
    assert resolve_warmup_steps(1000, 0.03, 5) == 5
    assert resolve_warmup_steps(10, 0.5, 100) == 10


def test_byte_tokenizer_roundtrip():
    tokenizer = ByteTokenizer()
    text = "State: x = 6 ✓"
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert tokenizer.decode(ids) == text
