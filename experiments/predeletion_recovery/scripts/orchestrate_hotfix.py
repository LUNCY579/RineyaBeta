#!/usr/bin/env python3
"""Run the PDRR matrix through the corrected v0.3 trainer entry point.

The original orchestrator remains preserved for provenance.  This wrapper
fail-closes unless it can replace every reference to ``train_hf.py`` with
``train_hf_hotfix.py`` in the current orchestrator source.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType


HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE / "orchestrate.py"


def build_patched_source() -> str:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    occurrences = source.count("train_hf.py")
    if occurrences < 1:
        raise RuntimeError(
            "orchestrator contains no train_hf.py reference; refusing to guess "
            "how to redirect the experiment matrix")
    corrected = source.replace("train_hf.py", "train_hf_hotfix.py")
    if "train_hf.py" in corrected:
        raise RuntimeError("not every trainer reference was redirected")
    return corrected


def main() -> None:
    source = build_patched_source()
    module = ModuleType("pdrr_orchestrate_v03_corrected")
    module.__file__ = str(SOURCE_PATH)
    module.__package__ = None
    namespace = module.__dict__
    exec(compile(source, str(SOURCE_PATH), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
