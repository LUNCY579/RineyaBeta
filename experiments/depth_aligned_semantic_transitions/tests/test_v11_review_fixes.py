"""Regression tests added after the adversarial review of the V1.1 repair commit.

The review found one real regression (multi-number answers scored by their first
number only) and several repairs that the first test pass left unpinned. Each test
here fails if the corresponding repair is reverted.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml

from dast.evaluation import answers_equal
from dast.model import DASTModel
from dast.toy import ToyCausalLM
from dast.training import _encode_targets, _run_status, set_global_seed
from tests.test_v11_repairs import _config, _encoder, _toy_model

PKG_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- the regression
def test_multi_number_golds_must_match_elementwise():
    """Comparing only the first number credited a wrong minute field on every
    HH:MM answer and a wrong range on every 'mean X, range Y' answer."""

    assert answers_equal("16:20", "16:55", "time") is False
    assert answers_equal("14:35", "14:05", "time") is False
    assert answers_equal("16:20", "16:20", "time") is True
    assert answers_equal("mean 3.5, range 12", "mean 3.5, range 99", "statistic") is False
    assert answers_equal("mean 3.5, range 12", "mean 3.5, range 12", "statistic") is True


def test_single_number_golds_keep_their_unit_tolerance():
    """The elementwise rule must not break the tolerance it was added alongside."""

    assert answers_equal("129 N", "129", "measurement") is True
    assert answers_equal("246 bolts", "246", "quantity") is True
    assert answers_equal("3 hours", "3", "rate") is True
    assert answers_equal("$14.50", "14.50", "money") is True
    assert answers_equal("0.25", "0.250", "decimal") is True
    assert answers_equal("7/2", "3.5", "fraction") is True
    assert answers_equal("129 N", "131", "measurement") is False


def test_number_count_mismatch_is_not_credited():
    assert answers_equal("mean 3.5, range 12", "3.5", "statistic") is False


# ------------------------------------------------------- left-padded generation
def test_generation_left_pads_and_builds_the_mask_from_pad_counts():
    """Nothing covered claim 4 before. Right padding, or a value-comparison mask,
    must fail here."""

    from dast import evaluation

    source = inspect.getsource(evaluation.evaluate_generation)
    assert "[pad_id] * (width - len(row)) + row" in source
    assert "row + [pad_id] * (width - len(row))" not in source
    # A value-comparison mask also blanks real tokens when pad_token_id == eos.
    assert "input_ids.ne(pad_id)" not in source
    assert "torch.ones_like(input_ids)" in source


def test_left_padded_layout_is_numerically_correct_for_ragged_batches():
    rows = [[5, 6, 7], [8], [9, 10]]
    width = max(len(row) for row in rows)
    pad = 0
    ids = torch.tensor([[pad] * (width - len(row)) + row for row in rows])
    mask = torch.ones_like(ids)
    for index, row in enumerate(rows):
        mask[index, : width - len(row)] = 0
    assert mask.tolist() == [[1, 1, 1], [0, 0, 1], [0, 1, 1]]
    # Every real token sits at the END, so generation continues from a real token
    # and a single decode offset of `width` is correct for every row.
    for index, row in enumerate(rows):
        assert ids[index, width - len(row):].tolist() == row


# ------------------------------------------------------------------- seed order
def test_seed_is_set_before_the_model_is_constructed():
    from dast import cli

    source = inspect.getsource(cli.command_train)
    seed_at = source.index("set_global_seed(")
    assert seed_at < source.index("build_hf_components("), "seed must precede LoRA init"
    assert seed_at < source.index("DASTModel("), "seed must precede projector init"


def test_seeding_makes_projector_init_reproducible():
    set_global_seed(17)
    first = _toy_model("raw")
    set_global_seed(17)
    second = _toy_model("raw")
    pairs = zip(first.state_projector.named_parameters(), second.state_projector.named_parameters())
    for (name, a), (_, b) in pairs:
        assert torch.equal(a, b), f"{name} differs across identically seeded builds"


# ------------------------------------------------------------- feedback readout
def test_feedback_hook_reads_the_same_tensor_as_the_auxiliary_path():
    """The hook saw the pre-norm layer output while the auxiliary path read the
    post-norm state, so the Phase-3 probe was fed a different vector than training."""

    from dast import model as model_module

    source = inspect.getsource(model_module.DASTModel._install_feedback_hooks)
    assert "_final_norm" in source

    ids = torch.tensor([[1, 2, 3, 4, 5]])
    mask = torch.ones_like(ids)
    base = ToyCausalLM(vocab_size=64, hidden_size=16, num_hidden_layers=8)
    common = dict(
        checkpoint_fractions=(0.25, 0.5, 0.75, 0.9), state_dim=8,
        projector_hidden_dim=16, pooling="mean", checkpoint_readout="final_norm",
    )
    aux = DASTModel(base, **common)
    fb = DASTModel(base, feedback_enabled=True, **common)
    fb.state_projector.load_state_dict(aux.state_projector.state_dict())
    with torch.no_grad():
        a = aux(input_ids=ids, attention_mask=mask, labels=None, prompt_mask=mask).states
        b = fb(input_ids=ids, attention_mask=mask, labels=None, prompt_mask=mask).states
    # The decoder is zero-initialised so the injected delta is zero; the two paths
    # must therefore agree exactly. They previously did not.
    assert torch.allclose(a, b, atol=1e-5)


def test_shipped_feedback_init_makes_every_intervention_a_no_op():
    """The fact the smoke used to hide by overwriting the decoder and the gates."""

    from dast.interventions import negate_state

    ids = torch.tensor([[1, 2, 3, 4, 5]])
    mask = torch.ones_like(ids)
    model = DASTModel(
        ToyCausalLM(vocab_size=64, hidden_size=16, num_hidden_layers=8),
        checkpoint_fractions=(0.25, 0.5, 0.75, 0.9), state_dim=8,
        projector_hidden_dim=16, pooling="mean", feedback_enabled=True,
    )
    model.eval()
    kwargs = dict(input_ids=ids, attention_mask=mask, labels=None, prompt_mask=mask)
    with torch.no_grad():
        plain = model(**kwargs)
        patched = model(**kwargs, interventions={1: negate_state})
        assert float((plain.logits - patched.logits).abs().max()) == 0.0
        torch.nn.init.normal_(model.feedback_decoder.weight, std=0.02)
        model.feedback_gates.fill_(2.0)
        live = model(**kwargs, interventions={1: negate_state})
    assert float((plain.logits - live.logits).abs().max()) > 0.0


# ------------------------------------------------------------- smoke, not blind
def test_smoke_projector_comparison_is_device_safe():
    from dast import smoke

    assert "parameter.detach().to(before[name].device)" in inspect.getsource(smoke.run_smoke)


def test_smoke_config_exercises_the_corrected_geometry():
    """The only end-to-end offline test must cover both protocol corrections, or CI
    is blind to them exactly as it was to the post-norm asymmetry."""

    from dast.smoke import build_smoke_config

    config = build_smoke_config(PKG_ROOT / "runs" / "_unit_test_only")
    assert config.model.checkpoint_readout == "final_norm"
    assert config.semantic.target_centering == "within_problem"
    assert config.model.checkpoint_fractions[-1] < 1.0


# --------------------------------------------------------- centering discipline
def test_centering_refuses_the_arms_own_grid_as_its_reference():
    with pytest.raises(ValueError, match="requires reference_texts"):
        _encode_targets(
            _encoder(), [["a x", "b y"]], device=torch.device("cpu"),
            centering="within_problem",
        )


def test_every_centering_call_site_passes_the_canonical_grid():
    from dast import training

    for function in (training.train, training.evaluate_teacher_forced):
        source = inspect.getsource(function)
        for chunk in source.split("_encode_targets(")[1:]:
            head = chunk[:400]
            if "centering=" in head:
                assert 'reference_texts=batch["canonical_target_texts"]' in head


# ------------------------------------------------------------------- governance
def test_sealed_train_split_is_visible_in_the_run_status():
    config = _config(path="/all.jsonl")
    config.data.train_split = "test"
    assert "SEALED_TEST" in _run_status(config, finished=True)


@pytest.mark.parametrize(
    "section,field,value,pattern",
    [
        ("experiment", "evidence_status", "exploratory", "evidence_status must be"),
        ("model", "checkpoint_readout", "finalnorm", "checkpoint_readout must be"),
        ("semantic", "target_centering", "within-problem", "target_centering must be"),
    ],
)
def test_new_literal_fields_are_validated_by_value(section, field, value, pattern):
    """A typo previously survived config load and surfaced as a bare KeyError after
    the model and corpus were already in memory."""

    config = _config(path="/all.jsonl")
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match=pattern):
        config.validate()


def test_confirmatory_dev_requires_a_frozen_preregistration():
    config = _config(path="/all.jsonl")
    config.experiment.evidence_status = "confirmatory_dev"
    with pytest.raises(ValueError, match="preregistration_frozen"):
        config.validate()
    config.experiment.preregistration_frozen = True
    config.validate()


def test_arm_configs_and_the_rehearsal_do_not_share_an_evidence_tier():
    """Without this, all six DEV arm runs emitted the same manifest status as the
    16-problem seam rehearsal."""

    base = yaml.safe_load((PKG_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    rehearsal = yaml.safe_load(
        (PKG_ROOT / "configs" / "seam_rehearsal.yaml").read_text(encoding="utf-8")
    )
    assert base["experiment"]["evidence_status"] == "exploratory_dev"
    assert rehearsal["experiment"]["evidence_status"] == "engineering"
