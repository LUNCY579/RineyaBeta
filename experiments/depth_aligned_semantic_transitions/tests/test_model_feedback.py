import torch

from dast.config import ARM_SPECS
from dast.data import build_examples, load_records
from dast.interventions import negate_state, swap_batch_states
from dast.model import DASTModel
from dast.operators import LowRankOperatorBank
from dast.smoke import fixture_path
from dast.tokenization import CausalBatchCollator
from dast.toy import HashTokenizer, ToyCausalLM


def _batch():
    tokenizer = HashTokenizer(256)
    records = load_records(fixture_path(), split="dev", limit=2)
    examples = build_examples(records, arm=ARM_SPECS["state_ordered"], num_stages=4, seed=17)
    return tokenizer, CausalBatchCollator(tokenizer, system_prompt="Solve.", max_length=160)(examples)


def test_auxiliary_model_uses_expected_hidden_checkpoints():
    _, batch = _batch()
    base = ToyCausalLM(vocab_size=256, hidden_size=32, num_hidden_layers=8)
    model = DASTModel(
        base,
        checkpoint_fractions=(0.25, 0.5, 0.75, 1.0),
        state_dim=12,
        projector_hidden_dim=24,
        pooling="mean",
        operator_bank=LowRankOperatorBank(12, 4, 3),
    )
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        prompt_mask=batch["prompt_mask"],
    )
    assert output.checkpoint_layers == (2, 4, 6, 8)
    assert output.states.shape == (2, 4, 12)
    assert output.transition_predictions.shape == (2, 3, 12)


def test_feedback_intervention_changes_downstream_logits():
    _, batch = _batch()
    base = ToyCausalLM(vocab_size=256, hidden_size=32, num_hidden_layers=8)
    model = DASTModel(
        base,
        checkpoint_fractions=(0.25, 0.5, 0.75, 1.0),
        state_dim=12,
        projector_hidden_dim=24,
        pooling="mean",
        feedback_enabled=True,
        feedback_scale=0.2,
    )
    assert model.feedback_decoder is not None and model.feedback_gates is not None
    torch.nn.init.normal_(model.feedback_decoder.weight, std=0.1)
    with torch.no_grad():
        model.feedback_gates.fill_(2.0)
        baseline = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], prompt_mask=batch["prompt_mask"]
        )
        intervened = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], prompt_mask=batch["prompt_mask"], interventions={1: negate_state}
        )
    assert not torch.allclose(baseline.logits, intervened.logits)


def test_batch_swap_intervention_shape_contract():
    states = torch.randn(2, 5)
    swapped = swap_batch_states(torch.tensor([1, 0]))(states)
    assert torch.equal(swapped[0], states[1])
