import torch

from dast.pooling import pool_prompt_hidden
from dast.semantic import FrozenTextTargetEncoder, SharedStateProjector
from dast.toy import HashTokenizer, ToyCausalLM


def test_answer_token_perturbation_cannot_change_prompt_pool():
    hidden = torch.randn(2, 6, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    baseline = pool_prompt_hidden(hidden, mask, "mean")
    changed = hidden.clone()
    changed[~mask] = 1_000_000
    assert torch.allclose(baseline, pool_prompt_hidden(changed, mask, "mean"))


def test_last_prompt_pool_handles_different_lengths():
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    mask = torch.tensor([[1, 1, 0, 0, 0], [0, 1, 1, 1, 0]], dtype=torch.bool)
    pooled = pool_prompt_hidden(hidden, mask, "last_prompt")
    assert torch.equal(pooled[0], hidden[0, 1])
    assert torch.equal(pooled[1], hidden[1, 3])


def test_shared_projector_outputs_unit_vectors():
    projector = SharedStateProjector(12, 5, 16)
    output = projector(torch.randn(4, 12))
    assert output.shape == (4, 5)
    assert torch.allclose(output.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_frozen_text_encoder_is_deterministic_and_gradient_free():
    model = ToyCausalLM(vocab_size=256, hidden_size=16, num_hidden_layers=2)
    model.get_input_embeddings().weight.requires_grad_(False)
    tokenizer = HashTokenizer(256)
    first = FrozenTextTargetEncoder(
        model.get_input_embeddings(), tokenizer, state_dim=8, projection_seed=3
    ).encode(["same meaning", "different"], device="cpu")
    second = FrozenTextTargetEncoder(
        model.get_input_embeddings(), tokenizer, state_dim=8, projection_seed=3
    ).encode(["same meaning", "different"], device="cpu")
    assert torch.equal(first, second)
    assert not first.requires_grad
    assert torch.allclose(first.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_target_encoder_rejects_trainable_embedding():
    model = ToyCausalLM(vocab_size=128, hidden_size=8, num_hidden_layers=1)
    tokenizer = HashTokenizer(128)
    try:
        FrozenTextTargetEncoder(model.get_input_embeddings(), tokenizer, state_dim=4, projection_seed=1)
    except ValueError as exc:
        assert "target drift" in str(exc)
    else:
        raise AssertionError("trainable target embedding should be rejected")
