#!/usr/bin/env python3
"""Analytic storage and compute accounting for a CSGQR remnant.

Purpose: establish the *ceiling* on what the mechanism can possibly save before
any of it is measured, so that a measured result can be read against a number
that was fixed in advance. If the ceiling is below the measurement noise of the
card, then no amount of routing cleverness produces a systems win at this scope,
and that is worth knowing before spending GPU time rather than after.

Three quantities are kept strictly apart, because conflating them is the single
easiest way to overstate a compression result:

``logical parameters``
    How many weights exist. Quantization does **not** change this. A 4-bit MLP
    has exactly as many parameters as a bf16 one.
``packed bytes``
    How many bytes those weights occupy in the chosen format, including scales
    and zero points. This is what quantization actually reduces.
``resident bytes``
    What is really on the device, which is ``packed bytes`` only if the
    full-precision donor was genuinely dropped. ``bench_gpu.py`` checks this
    against ``torch.cuda.memory_allocated`` rather than trusting the arithmetic
    here.

FLOPs are counted as 2*m*k multiply-accumulates per matmul and reported per
token. They are an upper bound on what routing can save and are explicitly not
a latency claim -- on this card, 4-bit weights must be dequantized to bf16
before the matmul, so the routed path does arithmetic the dense path does not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ModelShape:
    """The dimensions that drive the accounting."""

    layers: int
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tied_embeddings: bool
    dtype_bits: int = 16

    @classmethod
    def from_config(cls, config: dict, dtype_bits: int = 16) -> "ModelShape":
        return cls(
            layers=int(config["num_hidden_layers"]),
            hidden=int(config["hidden_size"]),
            intermediate=int(config["intermediate_size"]),
            heads=int(config["num_attention_heads"]),
            kv_heads=int(config["num_key_value_heads"]),
            head_dim=int(config.get("head_dim",
                                    config["hidden_size"] // config["num_attention_heads"])),
            vocab=int(config["vocab_size"]),
            tied_embeddings=bool(config.get("tie_word_embeddings", False)),
            dtype_bits=dtype_bits)

    # -- parameter counts --------------------------------------------------
    def mlp_params(self) -> int:
        """gate + up + down, all bias-free."""
        return 3 * self.hidden * self.intermediate

    def attn_params(self) -> int:
        q = self.hidden * self.heads * self.head_dim
        k = self.hidden * self.kv_heads * self.head_dim
        v = k
        o = self.heads * self.head_dim * self.hidden
        # Qwen3 adds RMSNorm over q and k head dims.
        return q + k + v + o + 2 * self.head_dim

    def layer_params(self) -> int:
        # Two RMSNorms per block.
        return self.mlp_params() + self.attn_params() + 2 * self.hidden

    def embedding_params(self) -> int:
        return self.vocab * self.hidden * (1 if self.tied_embeddings else 2)

    def total_params(self) -> int:
        return self.layers * self.layer_params() + self.embedding_params() + self.hidden

    # -- FLOPs -------------------------------------------------------------
    def mlp_macs(self, tokens: int = 1) -> int:
        return 2 * tokens * self.mlp_params()

    def attn_proj_macs(self, tokens: int = 1) -> int:
        return 2 * tokens * (self.attn_params() - 2 * self.head_dim)

    def lm_head_macs(self, tokens: int = 1) -> int:
        return 2 * tokens * self.vocab * self.hidden

    def model_macs(self, tokens: int = 1, *, include_lm_head: bool = True) -> int:
        """Weight-matmul MACs per token, ignoring the O(T^2) attention scores.

        Attention score/value products are excluded because they depend on
        context length rather than on weights, and every arm here shares them
        exactly -- including them would shrink the reported *share* of the MLP
        by an amount that says nothing about CSGQR.
        """
        per_layer = self.mlp_macs(tokens) + self.attn_proj_macs(tokens)
        total = self.layers * per_layer
        return total + (self.lm_head_macs(tokens) if include_lm_head else 0)


def bits_to_bytes(count: int, bits: int) -> float:
    return count * bits / 8.0


def remnant_storage(shape: ModelShape, *, bits: int, block: int,
                    symmetric: bool) -> dict[str, float]:
    """Packed bytes for one MLP remnant, itemized, plus its share of the model."""
    params = shape.mlp_params()
    n_blocks_per_row = -(-shape.hidden // block)      # ceil
    rows = 3 * shape.intermediate                     # gate, up, down^T
    weight_bytes = bits_to_bytes(params, bits) if bits != 16 else \
        bits_to_bytes(params, shape.dtype_bits)
    scale_bytes = 0.0 if bits == 16 else rows * n_blocks_per_row * 2      # fp16
    zero_bytes = 0.0 if (bits == 16 or symmetric) else rows * n_blocks_per_row  # uint8
    total = weight_bytes + scale_bytes + zero_bytes

    dense = bits_to_bytes(params, shape.dtype_bits)
    model_bytes = bits_to_bytes(shape.total_params(), shape.dtype_bits)
    return {
        "logical_parameters": params,
        "weight_bytes": weight_bytes,
        "scale_bytes": scale_bytes,
        "zero_bytes": zero_bytes,
        "packed_bytes": total,
        "dense_bytes": dense,
        "compression_vs_dense": dense / total if total else float("inf"),
        "bytes_saved_vs_dense": dense - total,
        "model_bytes_dense": model_bytes,
        "share_of_model_params": params / shape.total_params(),
        "saving_as_share_of_model": (dense - total) / model_bytes,
    }


def routing_ceiling(shape: ModelShape, *, n_groups: int,
                    active_fraction: float) -> dict[str, float]:
    """Compute saving available from routing one MLP layer at ``rho``.

    Upper bound, and deliberately framed as a *share of the whole model*: a 75%
    reduction of one layer's MLP is a much smaller number once divided by the 28
    layers and the LM head that still run in full.
    """
    mlp = shape.mlp_macs(1)
    model = shape.model_macs(1)
    skipped = mlp * (1.0 - active_fraction)
    return {
        "n_groups": n_groups,
        "active_fraction": active_fraction,
        "mlp_macs_per_token": mlp,
        "model_macs_per_token": model,
        "mlp_share_of_model_macs": mlp / model,
        "macs_skipped_per_token": skipped,
        "model_macs_saved_fraction": skipped / model,
    }


def confidence_overhead(shape: ModelShape) -> dict[str, float]:
    """What the exact LM-head confidence projection costs, per gated token.

    This is the number that decides whether an exact-confidence gate can pay for
    itself: it is compared against the MLP MACs the gate is trying to avoid. A
    ratio above 1 means the monitor costs more than the thing it is gating, and
    no threshold can rescue that.
    """
    lm = shape.lm_head_macs(1)
    mlp = shape.mlp_macs(1)
    logits_bytes = shape.vocab * shape.dtype_bits / 8.0
    return {
        "lm_head_macs_per_token": lm,
        "mlp_macs_per_token": mlp,
        "confidence_cost_ratio": lm / mlp,
        "logits_bytes_per_position": logits_bytes,
        "logits_bytes_per_1000_positions": logits_bytes * 1000,
    }


def report(shape: ModelShape, *, n_groups: int = 24, block: int = 128) -> dict:
    specs = [(16, True), (8, True), (8, False), (4, True), (4, False),
             (3, False), (2, False)]
    storage = {}
    for bits, sym in specs:
        key = "fp16" if bits == 16 else \
            "int" + str(bits) + ("-sym" if sym else "-asym")
        storage[key] = remnant_storage(shape, bits=bits, block=block, symmetric=sym)
    return {
        "shape": {
            "layers": shape.layers, "hidden": shape.hidden,
            "intermediate": shape.intermediate, "vocab": shape.vocab,
            "tied_embeddings": shape.tied_embeddings,
            "dtype_bits": shape.dtype_bits,
            "total_parameters": shape.total_params(),
            "model_bytes_dense": bits_to_bytes(shape.total_params(), shape.dtype_bits),
            "mlp_parameters_per_layer": shape.mlp_params(),
        },
        "block": block,
        "n_groups": n_groups,
        "group_size": shape.intermediate // n_groups,
        "storage": storage,
        "routing_ceiling": {
            format(rho, ".3f"): routing_ceiling(shape, n_groups=n_groups,
                                                active_fraction=rho)
            for rho in (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
        },
        "confidence_overhead": confidence_overhead(shape),
    }


def main() -> None:
    cfg = json.loads((REPO / "artifacts" / "rineya-core-en32k-28L"
                      / "config.json").read_text(encoding="utf-8"))
    shape = ModelShape.from_config(cfg, dtype_bits=16)
    out = report(shape)

    mib = 1 << 20
    s = out["shape"]
    print("model : " + str(s["layers"]) + "L hidden=" + str(s["hidden"])
          + " inter=" + str(s["intermediate"]) + " vocab=" + str(s["vocab"]))
    print("        " + format(s["total_parameters"], ",") + " parameters, "
          + format(s["model_bytes_dense"] / mib, ",.1f") + " MiB at bf16")
    print("        one MLP layer = " + format(s["mlp_parameters_per_layer"], ",")
          + " parameters ("
          + format(100 * s["mlp_parameters_per_layer"] / s["total_parameters"], ".2f")
          + "% of the model)")
    print()
    print("storage for ONE MLP remnant (block=" + str(out["block"]) + "):")
    head = "  {:<12s} {:>10s} {:>9s} {:>7s} {:>12s} {:>13s}"
    print(head.format("format", "packed MiB", "vs dense", "ratio", "saved MiB",
                      "% of model"))
    for name, v in out["storage"].items():
        print("  {:<12s} {:>10.3f} {:>9.3f} {:>7.2f} {:>12.3f} {:>12.3f}%".format(
            name, v["packed_bytes"] / mib, v["dense_bytes"] / mib,
            v["compression_vs_dense"], v["bytes_saved_vs_dense"] / mib,
            100 * v["saving_as_share_of_model"]))
    print()
    print("compute ceiling from routing one MLP layer:")
    print("  {:>6s} {:>18s} {:>22s}".format("rho", "MLP MACs skipped", "% of model MACs saved"))
    for rho, v in out["routing_ceiling"].items():
        print("  {:>6s} {:>18,.0f} {:>21.3f}%".format(
            rho, v["macs_skipped_per_token"], 100 * v["model_macs_saved_fraction"]))
    c = out["confidence_overhead"]
    print()
    print("exact-confidence monitor cost, per gated token:")
    print("  LM head  " + format(c["lm_head_macs_per_token"], ",.0f") + " MACs")
    print("  MLP      " + format(c["mlp_macs_per_token"], ",.0f") + " MACs")
    print("  ratio    " + format(c["confidence_cost_ratio"], ".3f")
          + "x  <- the monitor costs this much of what it gates")
    print("  logits   " + format(c["logits_bytes_per_position"] / 1024, ",.1f")
          + " KiB per position ("
          + format(c["logits_bytes_per_1000_positions"] / mib, ",.1f")
          + " MiB per 1000)")

    dest = REPO / "experiments" / "csgqr" / "results" / "budget.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\nwrote " + str(dest.relative_to(REPO)))


if __name__ == "__main__":
    main()
