#!/usr/bin/env python3
"""Install a CSGQR remnant in place of one Qwen3 layer's MLP.

Integration shape
-----------------
``Qwen3DecoderLayer`` runs its MLP sub-block as::

    residual = hidden_states                       # (A) pre-remnant residual
    hidden_states = post_attention_layernorm(hidden_states)
    hidden_states = mlp(hidden_states)             # (B) what we replace
    hidden_states = residual + hidden_states

The remnant replaces (B), but confidence must be read from (A), which (B) never
sees. Rather than monkeypatching ``Qwen3DecoderLayer.forward`` -- which would
silently rot the next time ``transformers`` reorders that function -- this
registers a ``forward_pre_hook`` on ``post_attention_layernorm``, whose input
*is* (A) by construction. The hook fires immediately before the MLP call in the
same forward pass, so the value is always the right one for the tokens being
processed. If the hook has not fired, :class:`CSGQRMlp` raises instead of
falling back to the post-norm tensor, because a silent fallback would change
what "confidence" means without changing any number that looks wrong.

Per-token routing, honestly
---------------------------
Different tokens select different group sets, which cannot be one matmul.
Tokens are therefore bucketed by their selected set and one matmul runs per
distinct set. For a static ranking the set is determined by ``k`` alone, so
there are at most ``n_groups + 1`` buckets regardless of sequence length. For
per-state dynamic significance the sets genuinely differ per token and the
bucket count approaches the token count -- that is a real cost of the method,
not an artefact, and ``bench_gpu.py`` measures it rather than hiding it behind
the FLOP count.

Prefill is gated too. Skipping the remnant for a prefill position changes that
position's hidden state and therefore its keys and values for every later layer,
so gating only the decode positions would be a different method. Both are
gated, per token, on their own confidence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from confidence import AlignmentLens, ConfidenceHead, features_from_hidden
from quant import QuantSpec
from remnant import QuantizedMlpRemnant, RemnantConfig
from router import RoutingPolicy, RoutingTrace


class CSGQRMlp(nn.Module):
    """Drop-in replacement for ``Qwen3MLP`` implementing the gated remnant.

    ``utility_fn`` maps the pre-remnant residual of one token to a per-group
    predicted utility, for the dynamic-significance selector. When it is None
    the policy uses its static ranking, which is the static-versus-dynamic
    contrast the ablation needs.
    """

    def __init__(self, remnant: QuantizedMlpRemnant, head: ConfidenceHead | None,
                 policy: RoutingPolicy, *,
                 utility_fn: Callable[[Tensor], Tensor] | None = None,
                 gate_fn: Callable[[Tensor], Tensor] | None = None,
                 trace: RoutingTrace | None = None) -> None:
        super().__init__()
        self.remnant = remnant
        self.head = head
        self.policy = policy
        self.utility_fn = utility_fn
        self.gate_fn = gate_fn
        self.trace = trace
        self._residual: Tensor | None = None
        self._token_counter = 0

        # What this arm actually needs is read off the policy, so an arm cannot
        # pay for a monitor its own configuration does not ask for, and cannot
        # skip one it does.
        if policy.needs_lens and head is None:
            raise ValueError(
                "policy " + policy.name() + " gates on the exact lens feature '"
                + policy.feature + "' but no ConfidenceHead was supplied")
        if policy.needs_probe and gate_fn is None:
            raise ValueError(
                "policy " + policy.name() + " gates on '" + policy.feature
                + "', which comes from a trained hidden-state probe, but no "
                "gate_fn was supplied. Train one with train_router.py.")
        if policy.needs_group_utility and utility_fn is None:
            raise ValueError(
                "policy " + policy.name() + " uses the dynamic-significance "
                "selector, which needs a per-state group-utility predictor, but "
                "no utility_fn was supplied. Earlier revisions silently fell "
                "back to the static ranking here, which made the dynamic arm a "
                "duplicate of the static one while still calling itself "
                "'significance'. Use selector='static_topk' for the static arm, "
                "or pass a utility_fn from train_router.py.")

    # -- what a forward pass will consult ----------------------------------
    @property
    def reads_residual(self) -> bool:
        """Does this arm read the pre-remnant residual ``h``?

        Both the gate signal and the dynamic-significance predictor read ``h``,
        not the post-norm MLP input. Anything that reads it needs the hook.
        """
        return (self.policy.needs_gate_signal
                or self.policy.needs_group_utility)

    # -- residual capture --------------------------------------------------
    def set_residual(self, residual: Tensor) -> None:
        self._residual = residual

    def _take_residual(self, like: Tensor) -> Tensor:
        if self._residual is None:
            raise RuntimeError(
                "the pre-remnant residual was never captured. install_csgqr() "
                "registers a forward_pre_hook on post_attention_layernorm to "
                "supply it; without it, confidence would silently be read from "
                "the post-norm tensor, which is a different quantity.")
        r = self._residual
        self._residual = None
        if r.shape[:-1] != like.shape[:-1]:
            raise RuntimeError("captured residual shape " + str(tuple(r.shape))
                               + " does not match MLP input " + str(tuple(like.shape)))
        return r

    def _drop_residual(self) -> None:
        """Release the captured residual on paths that do not consult it.

        The hook fires on every forward regardless of policy, so without this an
        always-on or deleted arm would hold a reference to one activation tensor
        for the lifetime of the module. Small, but it is resident bytes appearing
        in exactly the arms whose memory numbers are supposed to be the clean
        references.
        """
        self._residual = None

    # -- execution ---------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        # Fast paths: k is fixed in advance and the group set is the same for
        # every token, so there is nothing to bucket and no signal to compute.
        if self.policy.mode == "none":
            self._drop_residual()
            self._record_uniform(x, [])
            return torch.zeros_like(x)
        if self.policy.mode == "all":
            self._drop_residual()
            self._record_uniform(x, list(range(self.remnant.n_groups)))
            return self.remnant.forward(x)
        if (self.policy.mode == "fixed_fraction"
                and self.policy.selector not in ("random", "significance")):
            # random varies per token; significance varies per *state*. Every
            # other selector resolves to one fixed group set at a fixed k.
            self._drop_residual()
            groups = self.policy.select(0.0)
            self._record_uniform(x, groups)
            return self.remnant.forward_groups(x, groups)

        residual = self._take_residual(x) if self.reads_residual else x
        flat_x = x.reshape(-1, x.shape[-1])
        flat_r = residual.reshape(-1, residual.shape[-1])
        n = flat_x.shape[0]

        # -- the "how much" signal ----------------------------------------
        if self.policy.is_synthetic_gate:
            # The matched random-token control. No monitor is consulted, which is
            # the point: it isolates how much of a gate's benefit is skill and
            # how much is simply having spent that much compute.
            conf = [self.policy.synthetic_confidence(self._token_counter + i)
                    for i in range(n)]
        elif self.policy.needs_probe:
            # The probe predicts *need*; the policy thresholds *maturity*. One
            # subtraction puts them in the same orientation so a single
            # three-level implementation serves the lens and the probe, and a
            # tau means the same kind of thing in both.
            need = self.gate_fn(flat_r).to(torch.float32).reshape(-1)
            conf = (1.0 - need).clamp(0.0, 1.0).tolist()
        elif self.policy.needs_lens:
            feats = features_from_hidden(self.head, flat_r)
            conf = feats[self.policy.feature].to(torch.float32).tolist()
        else:
            conf = [0.0] * n

        # -- the "which groups" signal ------------------------------------
        utilities = None
        if self.policy.needs_group_utility:
            utilities = self.utility_fn(flat_r)
            if utilities.shape[0] != n:
                raise RuntimeError(
                    "utility_fn returned " + str(utilities.shape[0]) + " rows for "
                    + str(n) + " positions")
            # One transfer for the whole batch rather than one per token. The
            # selection below is a 24-element argsort per token; doing it on the
            # GPU means a launch and a synchronising .tolist() per token, which
            # dominates the routing cost at prefill lengths.
            utilities = utilities.detach().to("cpu", torch.float64)

        buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for i in range(n):
            u = None if utilities is None else utilities[i]
            groups = self.policy.select(conf[i], utility=u,
                                        token_index=self._token_counter + i)
            key = tuple(sorted(groups))
            buckets[key].append(i)
            if self.trace is not None:
                self.trace.record(conf[i], groups)
        self._token_counter += n

        out = torch.zeros_like(flat_x)
        for key, rows in buckets.items():
            if not key:
                continue
            idx = torch.tensor(rows, device=flat_x.device, dtype=torch.long)
            out[idx] = self.remnant.forward_groups(flat_x[idx], list(key))
        return out.view_as(x)

    def _record_uniform(self, x: Tensor, groups: list[int]) -> None:
        if self.trace is None:
            return
        for _ in range(x.reshape(-1, x.shape[-1]).shape[0]):
            self.trace.record(float("nan"), groups)


@dataclass
class InstalledCSGQR:
    """Handle for an installed remnant, so a run can report and undo itself."""

    model: nn.Module
    layer: int
    module: CSGQRMlp
    original_mlp: nn.Module
    hook_handle: object
    donor_layer: int
    donor_freed: bool = False

    def remove(self) -> None:
        """Restore the original MLP.

        Only possible when the donor was kept resident. With
        ``keep_donor_resident=False`` the full-precision projections were
        deleted to make the memory measurement honest, and there is nothing to
        restore -- so this raises rather than reinstalling a hollow module whose
        forward would fail somewhere far away.
        """
        if self.donor_freed:
            raise RuntimeError(
                "cannot restore layer " + str(self.layer) + ": the full-precision "
                "donor was freed at install time (keep_donor_resident=False). "
                "Reload the model to get an unmodified copy.")
        self.hook_handle.remove()
        self.model.model.layers[self.layer].mlp = self.original_mlp

    def report(self) -> dict[str, object]:
        r = self.module.remnant
        packed = r.packed_bytes()
        return {
            "layer": self.layer,
            "donor_layer": self.donor_layer,
            "wrong_layer_donor": self.donor_layer != self.layer,
            "spec": r.config.spec.name(),
            "bits": r.config.spec.bits,
            "block": r.config.spec.block,
            "symmetric": r.config.spec.symmetric,
            "n_groups": r.n_groups,
            "group_size": r.intermediate // r.n_groups,
            "packed_bytes": packed,
            "dense_bytes": r.dense_bytes(),
            "compression_vs_dense": r.dense_bytes() / max(packed["total"], 1),
            "policy": self.module.policy.to_dict(),
            "alpha": r.alpha.tolist(),
        }


def install_csgqr(model: nn.Module, layer: int, *,
                  spec: QuantSpec = QuantSpec(4, 128, False),
                  n_groups: int = 24,
                  policy: RoutingPolicy | None = None,
                  donor_layer: int | None = None,
                  lens_rank: int = 0,
                  temperature: float = 1.0,
                  utility_fn: Callable[[Tensor], Tensor] | None = None,
                  gate_fn: Callable[[Tensor], Tensor] | None = None,
                  trace: RoutingTrace | None = None,
                  keep_donor_resident: bool = False) -> InstalledCSGQR:
    """Replace ``model.model.layers[layer].mlp`` with a gated quantized remnant.

    ``donor_layer`` defaults to ``layer``. Setting it elsewhere builds the
    remnant from a *different* layer's weights while installing it at ``layer``
    -- the wrong-layer control for whether preserved donor identity matters at a
    matched byte budget.

    ``keep_donor_resident=False`` drops the original full-precision MLP after
    quantizing. That is the honest configuration: leaving an unreferenced bf16
    copy alive would make every memory measurement meaningless while every
    quality number stayed identical, which is precisely the failure mode that is
    easy to ship by accident.
    """
    layers = model.model.layers
    if not 0 <= layer < len(layers):
        raise ValueError("layer " + str(layer) + " outside [0, " + str(len(layers)) + ")")
    donor = layer if donor_layer is None else donor_layer
    if not 0 <= donor < len(layers):
        raise ValueError("donor layer " + str(donor) + " out of range")

    target = layers[layer]
    source_mlp = layers[donor].mlp
    dtype = target.mlp.gate_proj.weight.dtype
    device = target.mlp.gate_proj.weight.device

    cfg = RemnantConfig(layer=layer, n_groups=n_groups, spec=spec, compute_dtype=dtype)
    remnant = QuantizedMlpRemnant(source_mlp, cfg).to(device)

    policy = policy or RoutingPolicy(n_groups=n_groups)
    if policy.n_groups != n_groups:
        raise ValueError("policy.n_groups " + str(policy.n_groups)
                         + " disagrees with remnant n_groups " + str(n_groups))

    # Only an arm whose policy actually gates on the exact vocabulary lens pays
    # for a ConfidenceHead. A probe-gated arm must not construct one, or it would
    # be charged the |V|/(3*d_ff) monitor cost it exists to avoid.
    head = None
    if policy.needs_lens:
        head = ConfidenceHead(
            final_norm=model.model.norm,
            lm_head=model.lm_head,
            lens=AlignmentLens(model.config.hidden_size, lens_rank).to(device=device),
            vocab_size=int(model.config.vocab_size),
            temperature=temperature)

    module = CSGQRMlp(remnant, head, policy, utility_fn=utility_fn,
                      gate_fn=gate_fn, trace=trace)
    original = target.mlp
    target.mlp = module

    handle = target.post_attention_layernorm.register_forward_pre_hook(
        lambda _mod, args: module.set_residual(args[0]))

    if not keep_donor_resident:
        # Free the bf16 donor. The remnant holds packed bytes only; the original
        # module is dropped so a memory measurement means what it says.
        del original.gate_proj, original.up_proj, original.down_proj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return InstalledCSGQR(model=model, layer=layer, module=module,
                          original_mlp=original, hook_handle=handle,
                          donor_layer=donor, donor_freed=not keep_donor_resident)


def delete_mlp(model: nn.Module, layer: int) -> InstalledCSGQR:
    """Exact MLP deletion: the sub-block contributes nothing.

    Implemented through the same code path as every other arm, with
    ``mode="none"`` and a 4-bit remnant that is never executed, so that the
    deletion baseline cannot differ from the routed arms by anything other than
    routing. The remnant's bytes are still resident, which is why the deletion
    arm's *quality* is a valid floor while its *memory* is not a valid floor --
    see ``budget.py`` for the storage accounting that separates the two.
    """
    return install_csgqr(model, layer, policy=RoutingPolicy(mode="none", n_groups=1),
                        n_groups=1)
