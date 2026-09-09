"""Asset/contract fixtures only: these tests cannot constitute pretrained evidence."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments.official_family_v1 import runtime as r


def config(modern=False):
    result = {**r.SHAPE, "architectures": ["Qwen3ForCausalLM"], "attention_bias": False,
              "attention_dropout": 0.0, "use_sliding_window": False, "sliding_window": None,
              "hidden_act": "silu", "rms_norm_eps": 1e-6}
    result.update({"rope_parameters": {"rope_type": "default", "rope_theta": 1000000}}
                  if modern else {"rope_theta": 1000000, "rope_scaling": None})
    return result


@pytest.mark.parametrize("name", ["supa-1.7.0", "nimbus-1.6.0", "halo-1.2.0"])
def test_exact_profiles(name):
    profile = r.get_profile(name)
    assert len(profile["weight_sha256"]) == 64
    assert profile["weight_bytes"] > 10**9
    assert profile["core_id"].startswith("rineya.core.")


@pytest.mark.parametrize("name", ["Supa", "supa-a1.1", "R2", "Qwen/Qwen3-0.6B", "halo-uncensored", "nimbius"])
def test_ambiguous_or_other_checkpoints_rejected(name):
    with pytest.raises(r.FamilyError):
        r.get_profile(name)


@pytest.mark.parametrize("modern", [False, True])
def test_config_translation_is_explicit_and_read_only(modern):
    raw = config(modern)
    original = copy.deepcopy(raw)
    normalized = r.normalize_config(raw)
    assert raw == original
    assert normalized["rope_theta"] == 1000000
    assert normalized["rope_scaling"] is None
    assert "rope_parameters" not in normalized


@pytest.mark.parametrize("key,value", [
    ("vocab_size", 32768), ("num_hidden_layers", 12), ("eos_token_id", 32744),
    ("tie_word_embeddings", False), ("auto_map", None), ("abstain_logit_bias", 0),
    ("quantization_config", {}), ("peft_config", {}), ("rope_theta", 10000),
    ("rope_scaling", {"factor": 4}), ("attention_dropout", 0.1),
    ("use_sliding_window", True), ("layer_types", ["full_attention"]),
    ("architectures", ["SupaA1ForCausalLM"]), ("hidden_size", 1024.0),
])
def test_incompatible_config_rejected(key, value):
    raw = config()
    raw[key] = value
    with pytest.raises(r.FamilyError):
        r.normalize_config(raw)


@pytest.mark.parametrize("modern", [
    {"rope_type": "yarn", "rope_theta": 1000000, "factor": 4},
    {"rope_type": "default", "rope_theta": 10000},
])
def test_unregistered_modern_rope_rejected(modern):
    raw = config(True)
    raw["rope_parameters"] = modern
    with pytest.raises(r.FamilyError):
        r.normalize_config(raw)


def test_conflicting_rope_rejected():
    raw = config(True)
    raw["rope_theta"] = 10000
    with pytest.raises(r.FamilyError, match="Conflicting"):
        r.normalize_config(raw)


class Tokenizer:
    eos_token_id = 151645
    pad_token_id = 151643
    def __init__(self):
        self.vocab = {"hello": 0, "<|endoftext|>": 151643, "<|im_end|>": 151645}
    def get_vocab(self):
        return self.vocab


def test_padded_embedding_rows_are_not_tokenizer_length():
    result = r.validate_tokenizer(Tokenizer(), SimpleNamespace(**r.SHAPE))
    assert result["entries"] < result["embedding_rows"]
    assert result["eos_token_ids"] == [151645]


@pytest.mark.parametrize("bad_id", [151936, -1, True, 0.5])
def test_tokenizer_bad_id_rejected(bad_id):
    tok = Tokenizer()
    tok.vocab["bad"] = bad_id
    with pytest.raises(r.FamilyError):
        r.validate_tokenizer(tok, SimpleNamespace(**r.SHAPE))


def test_tokenizer_wrong_eos_rejected():
    tok = Tokenizer()
    tok.eos_token_id = 32744
    with pytest.raises(r.FamilyError):
        r.validate_tokenizer(tok, SimpleNamespace(**r.SHAPE))


@pytest.fixture
def assets(tmp_path, monkeypatch):
    # Explicit tiny serialized test fixture; not official weights and not exposed by CLI.
    import torch
    from safetensors.torch import save_file
    root = tmp_path / "model"
    root.mkdir()
    save_file({"fixture": torch.ones(4)}, root / "model.safetensors")
    weight = r.hash_file(root / "model.safetensors")
    profile = {"name": "fixture", "core_id": "fixture-not-pretrained", "directory": "fixture",
               "weight_sha256": weight["sha256"], "weight_bytes": weight["bytes"],
               "reference_sha256": {}}
    monkeypatch.setattr(r, "get_profile", lambda name: copy.deepcopy(profile))
    for name, content in {
        "CORE.json": {"core_id": profile["core_id"], "checks": {"weight_sha256": weight["sha256"], "weight_bytes": weight["bytes"]}},
        "config.json": config(), "tokenizer_config.json": {"tokenizer_class": "Qwen2Tokenizer"},
        "tokenizer.json": {"fixture": True},
    }.items():
        (root / name).write_text(json.dumps(content))
    return root, profile


def test_inventory_does_not_qualify(assets):
    root, _ = assets
    result = r.inventory(root, "fixture")
    assert result["evidence_class"] == "ASSET_VERIFICATION_ONLY"
    assert result["licenses_training"] is False
    assert result["tokenizer_reference_hash_verified"] is False


@pytest.mark.parametrize("name", r.REQUIRED)
def test_missing_assets_fail_closed(assets, name):
    root, _ = assets
    (root / name).unlink()
    with pytest.raises(r.FamilyError, match="missing"):
        r.inventory(root, "fixture")


def test_lfs_pointer_rejected(assets):
    root, _ = assets
    (root / "model.safetensors").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:fake\n")
    with pytest.raises(r.FamilyError, match="LFS"):
        r.inventory(root, "fixture")


def test_wrong_weights_rejected(assets):
    root, _ = assets
    with (root / "model.safetensors").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(r.FamilyError, match="Weights"):
        r.inventory(root, "fixture")


def test_wrong_core_id_rejected(assets):
    root, _ = assets
    core = r.read_json(root / "CORE.json")
    core["core_id"] = "rineya.core.supa-a1"
    (root / "CORE.json").write_text(json.dumps(core))
    with pytest.raises(r.FamilyError, match="CORE.json"):
        r.inventory(root, "fixture")


@pytest.mark.parametrize("name", ["adapter_config.json", "SYSTEM.json", "model.safetensors.index.json"])
def test_other_artifact_forms_rejected(assets, name):
    root, _ = assets
    (root / name).write_text("{}")
    with pytest.raises(r.FamilyError):
        r.inventory(root, "fixture")


def test_tokenizer_drift_is_bound(assets):
    root, _ = assets
    first = r.inventory(root, "fixture")
    (root / "tokenizer.json").write_text('{"fixture": "changed"}')
    assert r.inventory(root, "fixture")["identity"] != first["identity"]


def test_reference_tokenizer_hash_is_enforced(assets):
    root, profile = assets
    profile["reference_sha256"]["tokenizer.json"] = "0" * 64
    with pytest.raises(r.FamilyError, match="reference hash"):
        r.inventory(root, "fixture")


def test_symlink_checkpoint_rejected(assets, tmp_path):
    root, _ = assets
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(r.FamilyError):
        r.inventory(link, "fixture")


@pytest.mark.parametrize("key", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_partial_state_load_rejected(key):
    with pytest.raises(r.FamilyError):
        r.check_loading_info({key: ["bad"]})


def test_clean_state_load():
    r.check_loading_info({"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []})


def qreceipt(stage="qualify"):
    bound = {"experiment": "query_refinement_official_family_v1",
             "protocol": {"training_seeds": [17, 29, 43], "smoke_steps": 25}, "model": "fixture"}
    checks = {}
    for arm in ("baseline", "adapter", "independent", "feedback"):
        checks[f"{arm}/native_zero_max_error"] = 0.0
        checks[f"{arm}/cache_max_error"] = 0.0
        if arm != "baseline":
            checks[f"{arm}/trainable_parameters"] = 2064
    receipt = {"schema": "rineya.family-receipt/1", "stage": stage, "status": "pass", "binding": bound,
               "evidence_class": "PRETRAINED_OFFICIAL_FAMILY_EXPERIMENT",
               "sealed_evaluation_authorized": False, "quality_claim_authorized": False,
               "licenses_next_stage": "smoke", "qualification": checks, "outputs": {}}
    return bound, receipt


def test_valid_qualification_receipt_structure():
    bound, receipt = qreceipt()
    r.require_receipt(r.seal_receipt(receipt), bound, "qualify")


@pytest.mark.parametrize("mutation", ["legacy", "model", "source", "no_checks", "wrong_stage", "capacity", "budget", "claim"])
def test_stale_or_incomplete_receipts_rejected(mutation):
    bound, receipt = qreceipt()
    receipt = copy.deepcopy(receipt)
    if mutation == "legacy": receipt["schema"] = "qrr.native-receipt/1"
    if mutation == "model": receipt["binding"]["model"] = "other-profile"
    if mutation == "source": receipt["binding"]["source"] = "changed"
    if mutation == "no_checks": receipt.pop("qualification")
    if mutation == "wrong_stage": receipt["stage"] = "smoke"
    if mutation == "capacity": receipt["qualification"]["feedback/trainable_parameters"] = 1
    if mutation == "budget": receipt["licenses_next_stage"] = "dev"
    if mutation == "claim": receipt["quality_claim_authorized"] = True
    with pytest.raises(r.FamilyError):
        r.require_receipt(r.seal_receipt(receipt), bound, "qualify")


def test_smoke_requires_training_evidence():
    bound, receipt = qreceipt("smoke")
    receipt["licenses_next_stage"] = "dev"
    with pytest.raises(r.FamilyError, match="smoke"):
        r.require_receipt(r.seal_receipt(receipt), bound, "smoke")


def test_output_tampering_rejected(tmp_path):
    bound, receipt = qreceipt()
    output = tmp_path / "checks.json"
    output.write_text("before")
    receipt["outputs"] = {"checks.json": r.hash_file(output)}
    sealed = r.seal_receipt(receipt)
    output.write_text("after")
    with pytest.raises(r.FamilyError, match="changed"):
        r.require_receipt(sealed, bound, "qualify", tmp_path / "receipt.json")


@pytest.mark.parametrize("path", ["../outside", "/absolute", "..\\outside", "receipt.json"])
def test_receipt_output_path_traversal_rejected(path):
    bound, receipt = qreceipt()
    receipt["outputs"] = {path: {}}
    with pytest.raises(r.FamilyError, match="Unsafe"):
        r.require_receipt(r.seal_receipt(receipt), bound, "qualify")


def test_output_must_not_mutate_checkpoint(tmp_path):
    root = tmp_path / "model"
    with pytest.raises(r.FamilyError):
        r.check_output(root / "out", root, tmp_path / "src")


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"x": 1, "x": 2}')
    with pytest.raises(r.FamilyError, match="Duplicate"):
        r.read_json(path)


def test_write_is_exclusive(tmp_path):
    path = tmp_path / "receipt.json"
    r.write_new(path, {"once": True})
    with pytest.raises(FileExistsError):
        r.write_new(path, {"twice": True})


def test_qrr_draft_cannot_authorize(tmp_path):
    package = Path(__file__).resolve().parents[2] / "composition_qrr_v1" / "native.py"
    if not package.exists():
        pytest.skip("QRR integration is tested only on its successor branch")
    from experiments.composition_qrr_v1.family_native import validate_contract
    with pytest.raises(r.FamilyError):
        validate_contract({"schema": "rineya.family-frozen-contract/1", "status": "UNREVIEWED"}, {})


def test_query_protocol_uses_selected_family():
    package = Path(__file__).resolve().parents[2] / "query_refinement_v1" / "pipeline.py"
    if not package.exists():
        pytest.skip("Query integration is tested only on its successor branch")
    from experiments.query_refinement_v1.family_pipeline import family_protocol
    for profile in ("supa-1.7.0", "nimbus-1.6.0", "halo-1.2.0"):
        protocol = family_protocol(profile)
        assert protocol["target_config"]["vocab_size"] == 151936
        assert protocol["layer"] == 14 and protocol["rank"] == 8
        assert protocol["smoke_steps"] == 25 and protocol["dev_steps"] == 100
        assert "target_path" not in protocol


def test_restore_is_pinned_selective_and_materialized(assets, tmp_path, monkeypatch):
    import huggingface_hub
    from experiments.official_family_v1 import __main__ as cli
    root, profile = assets
    reg = r.registry()
    prefix = profile['directory'] + '/'
    names = [*r.REQUIRED, 'modeling_untrusted.py']
    seen = []
    class Api:
        def list_repo_files(self, **kwargs):
            assert kwargs == {'repo_id': reg['backup_repository'],
                              'revision': reg['backup_revision'], 'token': True}
            return [prefix + name for name in names]
    def download(**kwargs):
        assert kwargs['repo_id'] == reg['backup_repository']
        assert kwargs['revision'] == reg['backup_revision'] and kwargs['token'] is True
        name = kwargs['filename'].removeprefix(prefix)
        seen.append(name)
        return root / name
    monkeypatch.setattr(cli, 'get_profile', lambda name: profile)
    monkeypatch.setattr(huggingface_hub, 'HfApi', Api)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    target = tmp_path / 'restored'
    result = cli.restore('fixture', target)
    assert set(seen) == set(r.REQUIRED)
    assert result['pretrained_model_executed'] is False
    assert result['licenses_training'] is False
    assert all(not p.is_symlink() for p in target.iterdir())
    with pytest.raises(r.FamilyError, match='new'):
        cli.restore('fixture', target)


def test_restore_rejects_ambiguous_backup(assets, tmp_path, monkeypatch):
    import huggingface_hub
    from experiments.official_family_v1 import __main__ as cli
    _, profile = assets
    class Api:
        def list_repo_files(self, **kwargs):
            key = profile['directory'] + '/config.json'
            return [key, 'Rineya_Published_Cores/' + key]
    monkeypatch.setattr(cli, 'get_profile', lambda name: profile)
    monkeypatch.setattr(huggingface_hub, 'HfApi', Api)
    with pytest.raises(r.FamilyError, match='exactly one'):
        cli.restore('fixture', tmp_path / 'restored')
