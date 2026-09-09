"""Tiny random native Qwen3 fixtures. NEVER pretrained family qualification."""
import json
from pathlib import Path
import pytest
import torch
from experiments.official_family_v1 import runtime as r


def native_classes():
    pytest.importorskip("transformers", reason="Pinned native runtime is installed in CI")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    return Qwen3Config, Qwen3ForCausalLM


def small_model():
    Config, Model = native_classes()
    torch.manual_seed(1729)
    config = Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    head_dim=8, max_position_embeddings=128, attention_dropout=0.0,
                    tie_word_embeddings=True, rope_theta=1000000)
    config._attn_implementation = "eager"
    return Model(config).float().eval()


def test_native_safe_roundtrip_and_tied_head(tmp_path):
    model = small_model()
    model.save_pretrained(tmp_path, safe_serialization=True)
    _, Model = native_classes()
    loaded, info = Model.from_pretrained(tmp_path, local_files_only=True,
        use_safetensors=True, torch_dtype=torch.float32, attn_implementation="eager",
        output_loading_info=True)
    r.check_loading_info(info)
    assert loaded.lm_head.weight.data_ptr() == loaded.model.embed_tokens.weight.data_ptr()
    ids = torch.tensor([[1, 7, 12, 3]])
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids).logits, model(ids).logits)


def test_native_missing_parameter_is_not_accepted(tmp_path):
    from safetensors.torch import load_file, save_file
    model = small_model()
    model.save_pretrained(tmp_path, safe_serialization=True)
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    key = "model.layers.0.self_attn.q_proj.weight"
    tensors.pop(key)
    save_file(tensors, path, metadata={"format": "pt"})
    _, Model = native_classes()
    _, info = Model.from_pretrained(tmp_path, local_files_only=True,
        use_safetensors=True, output_loading_info=True)
    assert key in info["missing_keys"]
    with pytest.raises(r.FamilyError):
        r.check_loading_info(info)


def test_default_rope_serialization_translation():
    Config, _ = native_classes()
    base = {**r.SHAPE, "architectures": ["Qwen3ForCausalLM"],
            "hidden_act": "silu", "rms_norm_eps": 1e-6}
    old = Config.from_dict(r.normalize_config({**base, "rope_theta": 1000000}))
    new = Config.from_dict(r.normalize_config({**base, "rope_parameters": {
        "rope_type": "default", "rope_theta": 1000000}}))
    assert old.rope_theta == new.rope_theta == 1000000
    assert old.rope_scaling is new.rope_scaling is None


def test_query_native_integration_on_its_branch():
    root = Path(__file__).resolve().parents[2] / "query_refinement_v1"
    if not (root / "pipeline.py").exists():
        pytest.skip("Other independent successor branch")
    from experiments.query_refinement_v1.pipeline import qualify
    from experiments.query_refinement_v1.core import Update
    assert sum(p.numel() for p in Update(128, 16, 8).parameters()) == 2064
    checks = qualify(small_model(), layer=1, rank=2)
    assert all(f"{arm}/cache_max_error" in checks
               for arm in ("baseline", "adapter", "independent", "feedback"))


def test_frozen_native_integration_on_its_branch():
    root = Path(__file__).resolve().parents[2] / "composition_qrr_v1"
    if not (root / "native.py").exists():
        pytest.skip("Other independent successor branch")
    from experiments.composition_qrr_v1.native import greedy_trace
    class Tokenizer:
        def encode(self, *args, **kwargs): return [2, 3]
        def decode(self, ids, **kwargs): return "fixture"
    trace = greedy_trace(small_model().requires_grad_(False), Tokenizer(),
                         [1, 7, 12], 4, [], check_parity=True)
    assert len(trace["steps"]) == 4
    assert all(row["parity_argmax_equal"] for row in trace["steps"])
