#!/usr/bin/env python3
"""The variant catalogue: every attention shape this experiment evaluates.

A deliberate asymmetry runs through this file and it is the experiment's most
important structural finding, so it is encoded in the types rather than left in
prose:

  * **Sliding-window variants change no weights.** They are pure config
    overrides -- `use_sliding_window`, `sliding_window`, `layer_types` -- applied
    at load time. Zero disk cost, zero training cost, instantly reversible.
  * **GQA variants change weights.** Merging KV heads is irreversible surgery
    that produces a new checkpoint and *requires a healing run* before it means
    anything.

So "which axis is better" is not only a quality question. One axis is free and
one costs a training run per rung. A window schedule that merely ties a merged
GQA rung on quality still wins outright on cost.

Usage:
    python src/variants.py                 # list the catalogue with cache costs
    python src/variants.py --check         # confirm each config builds
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shapes import Shape, gqa, swa_bottom_global, swa_interleaved  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
BASE_MODEL = "artifacts/rineya-core-en32k-28L"  # R2
N_LAYERS = 28


@dataclass(frozen=True)
class Variant:
    name: str
    shape: Shape
    # Config overrides applied at load time (sliding-window axis).
    config_overrides: dict
    # Target KV heads when weight surgery is required (GQA axis); None = none.
    surgery_kv_heads: int | None = None
    note: str = ""

    @property
    def needs_surgery(self) -> bool:
        return self.surgery_kv_heads is not None

    @property
    def needs_healing(self) -> bool:
        return self.needs_surgery


def _swa_overrides(window: int, layer_types: list[str]) -> dict:
    return {
        "use_sliding_window": True,
        "sliding_window": window,
        "layer_types": layer_types,
        # max_window_layers is ignored once layer_types is explicit, but keep it
        # consistent so the config is not self-contradictory on inspection.
        "max_window_layers": layer_types.index("sliding_attention")
        if "sliding_attention" in layer_types else N_LAYERS,
    }


def bottom_global_types(n_global: int, n_layers: int = N_LAYERS) -> list[str]:
    """Qwen3Config semantics: layers with index >= n_global slide."""
    return ["full_attention" if i < n_global else "sliding_attention" for i in range(n_layers)]


def interleaved_types(period: int, n_layers: int = N_LAYERS) -> list[str]:
    """Every `period`-th layer global (Gemma-style)."""
    return ["full_attention" if (i + 1) % period == 0 else "sliding_attention"
            for i in range(n_layers)]


def catalogue(windows: tuple[int, ...] = (128, 256, 512, 1024)) -> list[Variant]:
    out: list[Variant] = [
        Variant("base-8KV", gqa(8), {}, None, "R2 as-is; the reference for everything"),
    ]

    # --- axis A: window schedule (free, config only) ------------------------
    for w in windows:
        for n_global in (2, 4, 7):
            lt = bottom_global_types(n_global)
            out.append(Variant(
                f"swa-w{w}-g{n_global}",
                swa_bottom_global(w, n_global),
                _swa_overrides(w, lt), None,
                f"bottom {n_global} layers global, {N_LAYERS - n_global} sliding",
            ))
        lt = interleaved_types(4)
        out.append(Variant(
            f"swa-w{w}-every4",
            swa_interleaved(w, 4),
            _swa_overrides(w, lt), None,
            "every 4th layer global (Gemma-style interleave)",
        ))

    # --- axis B: KV-head count (weight surgery + healing) -------------------
    for g in (4, 2, 1):
        out.append(Variant(
            f"gqa-{g}KV", gqa(g), {}, g,
            f"mean-pooled 8->{g} KV heads; UNHEALED numbers are a damage floor",
        ))

    return out


def load_variant(variant: Variant, base_model: str = BASE_MODEL, dtype=None,
                 device: str = "cpu", surgery_dir: str | None = None):
    """Build the model for a variant. Import torch lazily so listing stays cheap."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    if variant.needs_surgery:
        if surgery_dir is None:
            raise ValueError(
                f"{variant.name} needs weight surgery; run gqa_surgery.py first "
                f"and pass surgery_dir"
            )
        path = REPO / surgery_dir
    else:
        path = REPO / base_model

    cfg = AutoConfig.from_pretrained(path)
    for k, v in variant.config_overrides.items():
        setattr(cfg, k, copy.deepcopy(v))
    # Qwen3Config nulls sliding_window unless use_sliding_window is set; assigning
    # attributes post-construction bypasses that logic, so enforce it here.
    if variant.config_overrides.get("use_sliding_window") and cfg.sliding_window is None:
        raise ValueError(f"{variant.name}: sliding_window is None despite the override")

    if dtype is None:
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, config=cfg, dtype=dtype)
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="build each config (no weights)")
    ap.add_argument("--length", type=int, default=32768)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    cat = catalogue()
    ref = gqa(8)
    n = args.length

    print(f"variant catalogue -- cache cost at {n:,} tokens, base {BASE_MODEL}\n")
    print(f"{'variant':22} {'cache MiB':>10} {'vs 8KV':>8} {'weights?':>9} "
          f"{'heal?':>6}  note")
    for v in cat:
        mib = v.shape.cache_bytes(n) / 2 ** 20
        red = ref.cache_bytes(n) / v.shape.cache_bytes(n)
        print(f"{v.name:22} {mib:>10.0f} {red:>7.1f}x "
              f"{'SURGERY' if v.needs_surgery else 'none':>9} "
              f"{'yes' if v.needs_healing else 'no':>6}  {v.note}")

    free = [v for v in cat if not v.needs_surgery and v.name != "base-8KV"]
    paid = [v for v in cat if v.needs_surgery]
    print(f"\n{len(free)} variants cost nothing but a config override; "
          f"{len(paid)} need surgery plus a healing run each.")

    if args.check:
        from transformers import AutoConfig
        print("\nconfig build check:")
        cfg0 = AutoConfig.from_pretrained(REPO / BASE_MODEL)
        for v in cat:
            cfg = copy.deepcopy(cfg0)
            for k, val in v.config_overrides.items():
                setattr(cfg, k, copy.deepcopy(val))
            lt = list(getattr(cfg, "layer_types", []))
            n_slide = sum(t == "sliding_attention" for t in lt)
            print(f"  {v.name:22} sliding_window={str(cfg.sliding_window):>6} "
                  f"sliding_layers={n_slide:>2}/{len(lt)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([
            {"name": v.name, "cache_bytes": v.shape.cache_bytes(n),
             "needs_surgery": v.needs_surgery, "overrides": v.config_overrides,
             "surgery_kv_heads": v.surgery_kv_heads, "note": v.note}
            for v in cat], indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
