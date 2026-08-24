from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corrective_trainer_applies_both_fail_closed_replacements():
    module = load_script("train_hf_hotfix.py")
    source = module.build_patched_source()
    compile(source, str(module.SOURCE_PATH), "exec")

    assert 'if arm.name == "full"' in source
    assert 'output / "export_full"' in source
    assert '"mode": "intact_gate_unwrap"' in source
    assert '"temporary_gate_deleted": True' in source
    assert "orthogonal_drift" in source
    assert "preserve_weight" in source
    assert "zero_map" not in source


def test_corrective_orchestrator_redirects_every_trainer_reference():
    module = load_script("orchestrate_hotfix.py")
    original = module.SOURCE_PATH.read_text(encoding="utf-8")
    corrected = module.build_patched_source()
    compile(corrected, str(module.SOURCE_PATH), "exec")

    assert original.count("train_hf.py") >= 1
    assert "train_hf_hotfix.py" in corrected
    assert "train_hf.py" not in corrected
