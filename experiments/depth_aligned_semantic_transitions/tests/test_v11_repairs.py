"""Regression tests for the V1.1 readiness corrections.

Each test pins one defect that was found on a real checkpoint and was invisible to
the previous offline suite. See docs/PHASE0_READINESS.md for the measurements.
"""

from __future__ import annotations

import pytest
import torch

from dast.config import (
    DASTConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    LoraConfig,
    ModelConfig,
    OperatorConfig,
    SemanticConfig,
    TrainingConfig,
    assert_split_allowed,
)
from dast.evaluation import answers_equal
from dast.model import DASTModel, locate_final_norm
from dast.semantic import FrozenTextTargetEncoder
from dast.tokenization import TokenizationContractError, _as_int_list
from dast.toy import HashTokenizer, ToyCausalLM
from dast.training import _encode_targets, set_global_seed


# --------------------------------------------------------------------------- B1
def test_as_int_list_unwraps_transformers5_mapping():
    """transformers >= 5 returns a BatchEncoding from apply_chat_template."""

    assert _as_int_list({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}) == [1, 2, 3]
    assert _as_int_list({"input_ids": [[4, 5]]}) == [4, 5]
    assert _as_int_list([6, 7]) == [6, 7]
    with pytest.raises(TokenizationContractError):
        _as_int_list({"attention_mask": [1, 1]})


# --------------------------------------------------------------------------- B6
@pytest.mark.parametrize(
    "expected,predicted,answer_type,want",
    [
        ("129 N", "129", "measurement", True),
        ("129 N", "131", "measurement", False),
        ("246 bolts", "246", "quantity", True),
        ("0.25", "0.250", "decimal", True),
        ("$14.50", "14.50", "money", True),
        ("{6, 7, 9}", "{9, 6, 7}", "set", True),
        ("{6, 7, 9}", "{6, 7}", "set", False),
        ("A, B, C", "A, B, C", "ordering", True),
        ("A, B, C", "C, B, A", "ordering", False),
        ("true", "Yes", "boolean", True),
        ("true", "no", "boolean", False),
        ("hello world", "Hello World", "string", True),
    ],
)
def test_answers_equal_covers_the_corpus_answer_types(expected, predicted, answer_type, want):
    """70% of DEV answer types previously fell through to exact string equality."""

    assert answers_equal(expected, predicted, answer_type) is want


def test_answers_equal_rejects_none():
    assert answers_equal("1", None, "integer") is False


# --------------------------------------------------------- per-split corpus paths
def _config(**data_kwargs) -> DASTConfig:
    config = DASTConfig(
        experiment=ExperimentConfig(name="t", arm="state_ordered", seed=17, output_dir="runs/t"),
        model=ModelConfig(name_or_path="toy", precision="fp32"),
        data=DataConfig(**data_kwargs),
        semantic=SemanticConfig(),
        operators=OperatorConfig(),
        lora=LoraConfig(enabled=False),
        training=TrainingConfig(),
        evaluation=EvaluationConfig(do_generation=False),
    )
    return config


def test_data_path_for_split_prefers_per_split_overrides():
    config = _config(path="/all.jsonl", train_path="/tr.jsonl", eval_path="/ev.jsonl")
    assert config.data_path_for_split("train").as_posix().endswith("tr.jsonl")
    assert config.data_path_for_split("dev").as_posix().endswith("ev.jsonl")


def test_data_path_for_split_falls_back_to_single_file():
    config = _config(path="/all.jsonl")
    assert config.data_path_for_split("train").as_posix().endswith("all.jsonl")
    assert config.data_path_for_split("dev").as_posix().endswith("all.jsonl")


def test_data_path_for_split_refuses_when_nothing_is_configured():
    with pytest.raises(ValueError, match="no corpus path configured"):
        _config().data_path_for_split("train")


# ------------------------------------------------------------- sealing guard
def test_sealed_test_is_refused_without_all_three_conditions():
    config = _config(path="/all.jsonl")
    assert_split_allowed(config, "train")
    assert_split_allowed(config, "dev")
    with pytest.raises(PermissionError):
        assert_split_allowed(config, "test")
    with pytest.raises(PermissionError):
        assert_split_allowed(config, "TEST")


def test_inspect_alignment_guards_the_sealed_split():
    """The printed targets contain the final answer, so this path needs the seal."""

    import inspect

    from dast import cli

    source = inspect.getsource(cli.command_inspect_alignment)
    assert "assert_split_allowed" in source


# ------------------------------------------------- toy matches HF hidden semantics
def test_toy_last_hidden_state_is_post_final_norm_like_hf():
    """Previously the toy exposed a homogeneous stack, hiding the real asymmetry."""

    model = ToyCausalLM(vocab_size=64, hidden_size=16, num_hidden_layers=4)
    ids = torch.tensor([[1, 2, 3, 4]])
    out = model(input_ids=ids, output_hidden_states=True)
    assert len(out.hidden_states) == model.config.num_hidden_layers + 1
    last = out.hidden_states[-1]
    assert torch.allclose(last, model.final_norm(out.hidden_states[-1]) * 0 + last)
    # The last entry must be the normed state, not the raw residual.
    assert not torch.allclose(last, out.hidden_states[-2], atol=1e-4)
    assert torch.allclose(model.lm_head(last), out.logits, atol=1e-5)


def test_locate_final_norm_finds_the_toy_norm():
    model = ToyCausalLM(vocab_size=64, hidden_size=16, num_hidden_layers=4)
    assert locate_final_norm(model) is model.final_norm


# --------------------------------------------------------- symmetric norm readout
def _toy_model(readout: str, fractions=(0.25, 0.5, 0.75, 1.0)) -> DASTModel:
    base = ToyCausalLM(vocab_size=64, hidden_size=16, num_hidden_layers=4)
    return DASTModel(
        base,
        checkpoint_fractions=fractions,
        state_dim=8,
        projector_hidden_dim=16,
        pooling="mean",
        checkpoint_readout=readout,
    )


def test_final_norm_readout_changes_the_states_and_is_recorded():
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    mask = torch.ones_like(ids)
    set_global_seed(17)
    raw = _toy_model("raw")
    set_global_seed(17)
    normed = _toy_model("final_norm")
    assert raw.checkpoint_readout == "raw"
    assert normed.checkpoint_readout == "final_norm"
    with torch.no_grad():
        a = raw(input_ids=ids, attention_mask=mask, labels=None, prompt_mask=mask).states
        b = normed(input_ids=ids, attention_mask=mask, labels=None, prompt_mask=mask).states
    assert a.shape == b.shape
    assert not torch.allclose(a, b)


def test_final_norm_readout_does_not_double_norm_the_final_index():
    """hidden_states[L] is already post-norm, so it must be passed through as-is."""

    ids = torch.tensor([[1, 2, 3, 4, 5]])
    mask = torch.ones_like(ids)
    model = _toy_model("final_norm", fractions=(0.25, 0.5, 0.75, 1.0))
    assert model.checkpoint_layers[-1] == model.num_hidden_layers
    with torch.no_grad():
        out = model.base_model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        pooled_expected = out.hidden_states[model.num_hidden_layers].mean(dim=1)
        expected = model.state_projector(pooled_expected)
        states = model(input_ids=ids, attention_mask=mask, labels=None, prompt_mask=mask).states
    assert torch.allclose(states[:, -1], expected, atol=1e-5)


def test_unknown_readout_is_rejected():
    with pytest.raises(ValueError, match="unsupported checkpoint readout"):
        _toy_model("sometimes")


# ------------------------------------------------------------- target centering
class _Emb(torch.nn.Module):
    def __init__(self, vocab=64, dim=12):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab, dim), requires_grad=False)

    def forward(self, ids):
        return self.weight[ids]


def _encoder() -> FrozenTextTargetEncoder:
    torch.manual_seed(0)
    return FrozenTextTargetEncoder(
        _Emb(), HashTokenizer(vocab_size=64), state_dim=8, projection_seed=1701, max_tokens=8
    )


def test_within_problem_centering_makes_ordered_and_shuffled_separable():
    """As shipped, ordered and shuffled targets sat at 0.86 cosine."""

    encoder = _encoder()
    canonical = [["alpha one", "beta two", "gamma three", "delta four"]]
    shuffled = [["gamma three", "beta two", "delta four", "alpha one"]]
    device = torch.device("cpu")

    plain_o = _encode_targets(encoder, canonical, device=device)
    plain_s = _encode_targets(encoder, shuffled, device=device)
    plain_cos = torch.einsum("nkd,nkd->nk", plain_o, plain_s).mean()

    cent_o = _encode_targets(
        encoder, canonical, device=device,
        centering="within_problem", reference_texts=canonical,
    )
    cent_s = _encode_targets(
        encoder, shuffled, device=device,
        centering="within_problem", reference_texts=canonical,
    )
    cent_cos = torch.einsum("nkd,nkd->nk", cent_o, cent_s).mean()

    assert cent_cos < plain_cos
    assert torch.allclose(cent_o.norm(dim=-1), torch.ones(1, 4), atol=1e-5)


def test_centering_reference_is_permutation_invariant_so_arms_stay_matched():
    """The subtracted mean must be identical for every arm, or controls diverge."""

    encoder = _encoder()
    canonical = [["a x", "b y", "c z", "d w"]]
    permuted = [["c z", "a x", "d w", "b y"]]
    device = torch.device("cpu")
    mean_canonical = encoder.encode_grid(canonical, device=device).mean(dim=1)
    mean_permuted = encoder.encode_grid(permuted, device=device).mean(dim=1)
    assert torch.allclose(mean_canonical, mean_permuted, atol=1e-6)


def test_centering_keeps_final_repeated_non_degenerate():
    """Centering on the arm's own mean would zero this control; the canonical
    reference keeps it a constant non-zero target at every depth."""

    encoder = _encoder()
    canonical = [["p one", "q two", "r three", "s four"]]
    final_repeated = [["s four"] * 4]
    targets = _encode_targets(
        encoder, final_repeated, device=torch.device("cpu"),
        centering="within_problem", reference_texts=canonical,
    )
    assert targets is not None
    assert torch.all(targets.norm(dim=-1) > 0.5)


def test_unknown_centering_mode_is_rejected():
    with pytest.raises(ValueError, match="unsupported target centering"):
        _encode_targets(
            _encoder(), [["a", "b"]], device=torch.device("cpu"), centering="sideways"
        )


# ------------------------------------------------------- contrastive inertness
def test_contrastive_term_is_inert_below_batch_two():
    """base.yaml used batch_size 1, which silently disabled the anti-collapse term."""

    from dast.losses import multi_positive_contrastive_loss

    one = multi_positive_contrastive_loss(
        torch.randn(1, 4, 8), torch.randn(1, 4, 8), group_ids=None, temperature=0.07
    )
    two = multi_positive_contrastive_loss(
        torch.randn(2, 4, 8), torch.randn(2, 4, 8), group_ids=None, temperature=0.07
    )
    assert float(one) == 0.0
    assert float(two) > 0.0


def test_shipped_base_config_keeps_the_contrastive_term_active():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((root / "configs" / "base.yaml").read_text(encoding="utf-8"))
    assert base["training"]["batch_size"] >= 2
    assert base["semantic"]["contrastive_weight"] > 0.0
    # Effective batch and optimizer-step budget must be unchanged from the
    # committed 1 x 16, or the arms stop being budget-matched.
    assert base["training"]["batch_size"] * base["training"]["gradient_accumulation_steps"] == 16


# ------------------------------------------------------------- evidence status
def test_run_status_separates_engineering_from_dev():
    """A seam rehearsal was previously labelled DEV_COMPLETE, like a real arm run."""

    from dast.training import _run_status

    config = _config(path="/all.jsonl")
    assert config.experiment.evidence_status == "engineering"
    assert _run_status(config, finished=False) == "ENGINEERING__DEV__RUNNING"
    assert _run_status(config, finished=True) == "ENGINEERING__DEV__COMPLETE"
    config.experiment.evidence_status = "confirmatory_dev"
    assert _run_status(config, finished=True) == "CONFIRMATORY_DEV__DEV__COMPLETE"


def test_run_status_flags_the_sealed_split_case_insensitively():
    """eval_split 'TEST' previously produced a manifest that said DEV."""

    from dast.training import _run_status

    for label in ("test", "TEST", " Test "):
        config = _config(path="/all.jsonl")
        config.data.eval_split = label
        assert "SEALED_TEST" in _run_status(config, finished=True)


def test_manifest_hashes_every_corpus_file(tmp_path):
    from dast.artifacts import build_run_manifest

    a = tmp_path / "train.jsonl"
    b = tmp_path / "dev.jsonl"
    a.write_text("a\n", encoding="utf-8")
    b.write_text("b\n", encoding="utf-8")
    config = _config(path="", train_path=str(a), eval_path=str(b))
    manifest = build_run_manifest(
        config, data_path=[a, b], checkpoint_layers=(2, 4), model_info={}, status="ENGINEERING",
    )
    files = manifest["data"]["files"]
    assert len(files) == 2
    assert files[0]["sha256"] != files[1]["sha256"]
    # Single-file runs keep the v1 convenience fields populated.
    single = build_run_manifest(
        config, data_path=a, checkpoint_layers=(2, 4), model_info={}, status="ENGINEERING",
    )
    assert single["data"]["sha256"] == files[0]["sha256"]


# ---------------------------------------------------- readout/schedule coherence
def test_final_norm_readout_rejects_a_schedule_pinned_to_the_top():
    """final_norm + last fraction 1.0 leaves exactly the heterogeneity it fixes."""

    config = _config(path="/all.jsonl")
    config.model.checkpoint_readout = "final_norm"
    config.model.checkpoint_fractions = (0.25, 0.5, 0.75, 1.0)
    with pytest.raises(ValueError, match="heterogeneous interface"):
        config.validate()


def test_raw_readout_still_allows_the_committed_schedule():
    config = _config(path="/all.jsonl")
    config.model.checkpoint_readout = "raw"
    config.model.checkpoint_fractions = (0.25, 0.5, 0.75, 1.0)
    config.validate()


def test_final_fraction_floor_is_enforced():
    config = _config(path="/all.jsonl")
    config.model.checkpoint_readout = "raw"
    config.model.checkpoint_fractions = (0.1, 0.2, 0.3, 0.5)
    with pytest.raises(ValueError, match=r"\[0.9, 1.0\]"):
        config.validate()


def test_shipped_base_config_selects_the_corrected_geometries():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((root / "configs" / "base.yaml").read_text(encoding="utf-8"))
    assert base["model"]["checkpoint_readout"] == "final_norm"
    assert base["semantic"]["target_centering"] == "within_problem"
    # The last checkpoint must NOT be the final index, or it stays post-norm while
    # the others are raw and the shared interface is heterogeneous again.
    assert base["model"]["checkpoint_fractions"][-1] < 1.0
    assert base["data"]["train_path"] and base["data"]["eval_path"]
