import torch

from dast.config import ARM_SPECS
from dast.data import build_examples, load_records
from dast.smoke import fixture_path
from dast.tokenization import CausalBatchCollator, IGNORE_INDEX, tokenize_example
from dast.toy import HashTokenizer


def _example():
    record = load_records(fixture_path(), split="train", limit=1)[0]
    return build_examples([record], arm=ARM_SPECS["state_ordered"], num_stages=4, seed=17)[0]


def test_prompt_mask_and_lm_labels_are_disjoint():
    item = tokenize_example(
        _example(),
        HashTokenizer(256),
        system_prompt="Solve.",
        max_length=256,
        truncation="error",
    )
    for prompt, label in zip(item.prompt_mask, item.labels):
        if prompt:
            assert label == IGNORE_INDEX
        else:
            assert label != IGNORE_INDEX


def test_collator_padding_never_enters_prompt_mask():
    tokenizer = HashTokenizer(256)
    records = load_records(fixture_path(), split="train", limit=2)
    examples = build_examples(records, arm=ARM_SPECS["state_ordered"], num_stages=4, seed=17)
    batch = CausalBatchCollator(tokenizer, system_prompt="Solve.", max_length=256)(examples)
    assert not torch.any(batch["prompt_mask"] & ~batch["attention_mask"].bool())
    assert batch["training_group_ids"].shape == (2, 4)


def test_preserve_target_truncation_keeps_assistant_tokens():
    item = tokenize_example(
        _example(),
        HashTokenizer(256),
        system_prompt="system " * 100,
        max_length=64,
        truncation="preserve_target",
    )
    assert item.truncated_prompt_tokens > 0
    assert any(label != IGNORE_INDEX for label in item.labels)
