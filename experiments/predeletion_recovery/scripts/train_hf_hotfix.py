#!/usr/bin/env python3
"""Corrective v0.3 entry point for the real-model PDRR trainer.

This wrapper deliberately transforms two narrowly identified blocks in the
committed ``train_hf.py`` before executing it:

1. the intact ``full`` anchor removes the temporary layer-gate wrapper without
   physically deleting the target block, verifies numerical identity, and
   exports to ``export_full``;
2. the recipient stability term permits acquired residual signal aligned with
   the doomed block's missing contribution while penalizing orthogonal drift,
   especially where the source intervention has low causal significance.

The transformation is fail-closed.  If the underlying trainer changes and either
expected block is no longer found exactly once, execution stops instead of
silently running an unpatched or partially patched experiment.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType


HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE / "train_hf.py"


OLD_ANCHOR = '''                    current_map = {layer: delta for layer, delta in zip(sorted(probes), deltas)}
                    zero_map = {layer: torch.zeros_like(delta) for layer, delta in current_map.items()}
                    anchor = anchor_preservation_loss(
                        current_map, zero_map, recipient_layers=sorted(probes))
'''

NEW_ANCHOR = '''                    # Preserve recipients' existing duties without forbidding
                    # acquisition of the missing job.  A plain ||delta||^2
                    # penalty classifies every useful handoff as damage.  Allow
                    # the acquired component aligned with the doomed block's
                    # missing residual, while penalizing orthogonal functional
                    # drift—most strongly where ablating the source mattered
                    # little for the teacher distribution.
                    source_direction = source_residual.detach().float()
                    source_energy = source_direction.square().sum(
                        dim=-1, keepdim=True).clamp_min(1e-8)
                    causal_weight = significance.detach().float()
                    causal_weight = causal_weight / causal_weight.amax(
                        dim=-1, keepdim=True).clamp_min(1e-8)
                    preserve_weight = (1.0 - causal_weight).clamp_min(0.05)
                    anchor_terms = []
                    for delta in deltas:
                        delta_f = delta.float()
                        projection = (
                            (delta_f * source_direction).sum(dim=-1, keepdim=True)
                            / source_energy
                        ) * source_direction
                        orthogonal_drift = (delta_f - projection).square().mean(dim=-1)
                        anchor_terms.append(
                            (orthogonal_drift * preserve_weight).sum()
                            / preserve_weight.sum().clamp_min(1.0)
                        )
                    anchor = torch.stack(anchor_terms).mean() if anchor_terms else zero
'''

OLD_EXPORT = '''    if last_sample is None:
        raise RuntimeError("training produced no sample for export verification")
    set_layer_gate(student, target, 0.0)
    set_probe_capture(probes, False)
    remove_recipient_probes(student, probes)
    compact, report = scar_free_check(student, last_sample, target)
    if not report["passed"]:
        raise RuntimeError(f"scar-free export check failed: {report}")
    export = output / "export_compact"
    compact.save_pretrained(export, safe_serialization=True)
    tokenizer.save_pretrained(export)
    atomic_write_json(output / "export_manifest.json", {
        "arm": arm.name,
        "target_layer": target,
        "recipients": recipients,
        "router_deleted": True,
        "temporary_probes_deleted": True,
        **report,
    })
    print(f"exported scar-free compact checkpoint to {export}")
'''

NEW_EXPORT = '''    if last_sample is None:
        raise RuntimeError("training produced no sample for export verification")
    set_probe_capture(probes, False)
    remove_recipient_probes(student, probes)

    if arm.name == "full":
        # The intact anchor must not retain the temporary gate wrapper and must
        # not lose the target block.  Verify that unwrapping is numerically inert.
        set_layer_gate(student, target, 1.0)
        student.eval()
        with torch.inference_mode():
            before = student(**last_sample).logits.float()
        layers = resolve_decoder_layers(student)
        wrapper = layers[target]
        inner = None
        for attribute in ("layer", "block", "wrapped", "module", "inner"):
            candidate = getattr(wrapper, attribute, None)
            if isinstance(candidate, torch.nn.Module):
                inner = candidate
                break
        if inner is None:
            raise RuntimeError(
                f"cannot unwrap temporary layer gate {type(wrapper).__name__}; "
                "refusing to export a contaminated full anchor")
        parameter_count_before = sum(
            parameter.numel() for parameter in student.parameters())
        layers[target] = inner
        with torch.inference_mode():
            after = student(**last_sample).logits.float()
        max_abs = float((before - after).abs().max())
        argmax_equal = bool(torch.equal(
            before.argmax(dim=-1), after.argmax(dim=-1)))
        parameter_count_after = sum(
            parameter.numel() for parameter in student.parameters())
        tolerance = 1e-5
        report = {
            "passed": (
                max_abs <= tolerance
                and argmax_equal
                and parameter_count_after == parameter_count_before
            ),
            "mode": "intact_gate_unwrap",
            "gate_zero_physical_max_abs": None,
            "unwrap_max_abs": max_abs,
            "greedy_argmax_equal": argmax_equal,
            "original_parameter_count": parameter_count_before,
            "final_parameter_count": parameter_count_after,
            "parameter_reduction": 0,
            "num_hidden_layers": len(resolve_decoder_layers(student)),
        }
        if not report["passed"]:
            raise RuntimeError(f"intact gate-unwrapping check failed: {report}")
        export = output / "export_full"
        deployment = student
    else:
        set_layer_gate(student, target, 0.0)
        deployment, report = scar_free_check(student, last_sample, target)
        if not report["passed"]:
            raise RuntimeError(f"scar-free export check failed: {report}")
        export = output / "export_compact"

    deployment.save_pretrained(export, safe_serialization=True)
    tokenizer.save_pretrained(export)
    atomic_write_json(output / "export_manifest.json", {
        "arm": arm.name,
        "target_layer": target,
        "recipients": recipients,
        "router_deleted": True,
        "temporary_probes_deleted": True,
        "temporary_gate_deleted": True,
        "export_kind": "full" if arm.name == "full" else "compact",
        **report,
    })
    print(
        f"exported verified "
        f"{'intact' if arm.name == 'full' else 'scar-free compact'} "
        f"checkpoint to {export}")
'''


def build_patched_source() -> str:
    """Return a fail-closed corrected trainer source string."""
    source = SOURCE_PATH.read_text(encoding="utf-8")
    if source.count(OLD_ANCHOR) != 1:
        raise RuntimeError(
            "expected exactly one legacy recipient-anchor block; "
            "the base trainer changed, so the hotfix refuses to guess")
    if source.count(OLD_EXPORT) != 1:
        raise RuntimeError(
            "expected exactly one legacy export block; "
            "the base trainer changed, so the hotfix refuses to guess")
    source = source.replace(OLD_ANCHOR, NEW_ANCHOR, 1)
    source = source.replace(OLD_EXPORT, NEW_EXPORT, 1)
    return source


def main() -> None:
    source = build_patched_source()
    module = ModuleType("pdrr_train_hf_v03_corrected")
    module.__file__ = str(SOURCE_PATH)
    module.__package__ = None
    namespace = module.__dict__
    exec(compile(source, str(SOURCE_PATH), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
