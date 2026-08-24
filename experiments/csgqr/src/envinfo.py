#!/usr/bin/env python3
"""Environment manifest for the CSGQR campaign.

Because this research targets consumer hardware, the machine is part of the
experiment rather than a footnote. Every result file in this experiment carries
the ``environment`` block this module produces, so a number can always be tied
to the card and the software stack that produced it.

Two traps this records deliberately:

  * The *ambient* interpreter on this machine has a CPU-only torch
    (``2.13.0+cpu``). Only ``.venv/Scripts/python.exe`` has ``+cu130``. A run
    that silently landed on the CPU build would still produce quality numbers,
    and they would be indistinguishable from GPU numbers in a summary file --
    so ``torch_cuda_build`` and ``cuda_available`` are recorded, and
    ``require_cuda`` fails loudly rather than falling back.
  * Windows/WDDM spills past VRAM into host RAM instead of raising, turning an
    over-budget run into a ~100x slowdown rather than an error (see
    scripts/train/memory.py). ``total_vram_mib`` and the free-memory reading at
    process start are recorded so a spill is diagnosable after the fact.

Usage:
    python experiments/csgqr/src/envinfo.py            # print + write manifest
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
MIB = 1 << 20


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(("git", *args), cwd=REPO, capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _nvidia_smi() -> dict[str, Any] | None:
    """Driver-level view, as a cross-check on what torch reports."""
    fields = ("name", "memory.total", "memory.used", "memory.free",
              "driver_version", "compute_cap")
    query = "--query-gpu=" + ",".join(fields)
    try:
        out = subprocess.run(("nvidia-smi", query, "--format=csv,noheader,nounits"),
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
    if len(parts) != len(fields):
        return None
    return dict(zip(fields, parts))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def model_identity(model_dir: Path) -> dict[str, Any]:
    """Identity of the checkpoint under test.

    The weights file is ~948 MB; hashing it takes a few seconds and is worth it,
    because "which checkpoint produced this number" is exactly the question a
    reader of a results file cannot otherwise answer.
    """
    weights = model_dir / "model.safetensors"
    config = model_dir / "config.json"
    out: dict[str, Any] = {"dir": model_dir.name}
    if config.exists():
        out["config_sha256"] = sha256_file(config)
        out["config"] = json.loads(config.read_text(encoding="utf-8"))
    if weights.exists():
        out["weights_bytes"] = weights.stat().st_size
        out["weights_sha256"] = sha256_file(weights)
    return out


def collect(*, with_model: Path | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }

    try:
        import torch
    except ImportError:
        info["torch"] = None
        return info

    info["torch"] = torch.__version__
    info["torch_cuda_build"] = torch.version.cuda
    info["cuda_available"] = bool(torch.cuda.is_available())
    try:
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001 - absent cudnn is data, not an error
        info["cudnn"] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        info["gpu"] = {
            "name": props.name,
            "capability": str(props.major) + "." + str(props.minor),
            "total_vram_mib": round(props.total_memory / MIB, 1),
            "multi_processor_count": props.multi_processor_count,
            "free_mib_at_start": round(free / MIB, 1),
            "used_by_others_mib": round((total - free) / MIB, 1),
        }
    info["nvidia_smi"] = _nvidia_smi()

    for name in ("transformers", "peft", "accelerate", "numpy", "scipy",
                 "datasets", "safetensors"):
        try:
            info[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - a missing optional dep is data, not an error
            info[name] = None

    if with_model is not None:
        info["model"] = model_identity(with_model)
    return info


def require_cuda() -> None:
    """Fail loudly when a GPU run would silently land on the CPU build."""
    import torch
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to this interpreter.\n"
            "  executable : " + sys.executable + "\n"
            "  torch      : " + torch.__version__ +
            " (cuda build: " + str(torch.version.cuda) + ")\n"
            "The ambient python on this machine carries a CPU-only torch. Use\n"
            "  .venv/Scripts/python.exe\n"
            "A CPU fallback would produce quality numbers that look identical to\n"
            "GPU numbers in a summary file, so this is refused rather than warned.")


def main() -> None:
    model = REPO / "artifacts" / "rineya-core-en32k-28L"
    info = collect(with_model=model if model.exists() else None)
    out = REPO / "experiments" / "csgqr" / "results" / "environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("python        " + info["python"] + "  (" + info["executable"] + ")")
    print("platform      " + info["platform"])
    print("torch         " + str(info.get("torch")) + "  cuda build "
          + str(info.get("torch_cuda_build")) + "  available="
          + str(info.get("cuda_available")))
    print("transformers  " + str(info.get("transformers")) + "   peft "
          + str(info.get("peft")))
    gpu = info.get("gpu")
    if gpu:
        print("gpu           " + gpu["name"] + "  cc" + gpu["capability"] + "  "
              + format(gpu["total_vram_mib"], ",.0f") + " MiB total, "
              + format(gpu["free_mib_at_start"], ",.0f") + " MiB free at start")
    smi = info.get("nvidia_smi")
    if smi:
        print("driver        " + smi["driver_version"] + "  (smi reports "
              + smi["memory.total"] + " MiB total, " + smi["memory.used"]
              + " MiB used)")
    m = info.get("model")
    if m:
        cfg = m.get("config", {})
        print("model         " + m["dir"] + "  " + str(cfg.get("num_hidden_layers"))
              + "L hidden=" + str(cfg.get("hidden_size"))
              + " inter=" + str(cfg.get("intermediate_size"))
              + " vocab=" + str(cfg.get("vocab_size"))
              + " tied=" + str(cfg.get("tie_word_embeddings")))
        print("weights       " + format(m.get("weights_bytes", 0), ",")
              + " bytes  sha256 " + str(m.get("weights_sha256"))[:16] + "...")
    print("git           " + str(info.get("git_branch")) + " @ "
          + str(info.get("git_sha"))[:12]
          + ("  (DIRTY)" if info.get("git_dirty") else ""))
    print("\nwrote " + str(out.relative_to(REPO)))


if __name__ == "__main__":
    main()
