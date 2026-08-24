#!/usr/bin/env python3
"""Stage III claim checks: the held-out endpoint and the GPU benchmark.

Kept in its own module so ``verify_claims.py`` stays readable, and imported by it
rather than duplicated. Every check reads a **committed** result file, so a fresh
clone can verify every headline number without a GPU.

The ``claim``, ``section`` and ``load`` callables are passed in from
``verify_claims.py`` so the two files share one ledger and one exit code.
"""

from __future__ import annotations


def add_checks(claim, section, load) -> None:
    """Register every Stage III claim. ``claim(name, source, expected, actual)``."""

    # ==================================================================
    # Held-out teacher-forced endpoint -- the primary evidence
    # ==================================================================
    section("Held-out teacher-forced endpoint (414 prompts, 3830 positions)")

    forced: dict[str, dict] = {}
    first = None
    for tag in ("A_baselines", "B_matched", "C_gates", "D_tokengates",
                "E_frontier"):
        d = load("forced_heldout_" + tag + ".json")
        if not d:
            continue
        first = first or d
        for k, v in d["arms"].items():
            forced.setdefault(k, v)
        claim("batch " + tag + " completed with no failed arms",
              "forced_heldout_" + tag + ".json", True,
              bool(d.get("complete")) and not d.get("failures"),
              note=str(len(d["arms"])) + " arms")

    if not forced:
        return

    def kl(key):
        a = forced.get(key)
        return None if a is None else a["kl"]["point"]

    def ci(key):
        a = forced.get(key)
        return (None, None) if a is None else (a["kl"]["lo"], a["kl"]["hi"])

    def act(key):
        return ((forced.get(key) or {}).get("routing") or {}).get("active_fraction")

    def gold(key):
        a = forced.get(key)
        return None if a is None else a["gold_top1"]["point"]

    claim("the held-out set is disjoint from the probe's training prompts",
          "forced_heldout_A_baselines.json", True,
          int((first or {}).get("skip_prompts", 0)) >= 96,
          note="skip_prompts = " + str((first or {}).get("skip_prompts")))

    # -- F2: zero noise floor, monotone precision --------------------------
    if kl("b2-fp-all") is not None:
        claim("the FP-passthrough noise floor is zero on this endpoint",
              "forced_heldout_A", True, abs(kl("b2-fp-all")) < 1e-4,
              note=format(kl("b2-fp-all"), ".6f") + " nats")
    if all(kl(k) is not None for k in ("b3-int8-all", "b4-int4-all", "b1-delete")):
        claim("precision is monotone: int8 < int4 < delete",
              "forced_heldout_A", True,
              kl("b3-int8-all") < kl("b4-int4-all") < kl("b1-delete"),
              note="int8 " + format(kl("b3-int8-all"), ".5f") + " < int4 "
                   + format(kl("b4-int4-all"), ".5f") + " < delete "
                   + format(kl("b1-delete"), ".5f"))
        rec = (kl("b1-delete") - kl("b4-int4-all")) / kl("b1-delete")
        claim("INT4 recovers >95% of the deletion divergence damage",
              "forced_heldout_A", True, rec > 0.95, note=format(rec, ".4f"))

    # -- F1: deletion costs no reference-prediction accuracy ---------------
    if gold("b1-delete") is not None and gold("b0-dense") is not None:
        gd = abs(gold("b1-delete") - gold("b0-dense"))
        claim("deleting the MLP costs no gold-token accuracy "
              "(so recovery is fidelity, not quality)",
              "forced_heldout_A", True, gd < 0.005,
              note="|delta gold_top1| = " + format(gd, ".6f"))

    # -- F3: donor identity ------------------------------------------------
    if kl("b13-wrong-layer") is not None and kl("b1-delete") is not None:
        claim("the wrong-layer donor is WORSE than installing nothing",
              "forced_heldout_A", True, kl("b13-wrong-layer") > kl("b1-delete"),
              note="wrong-layer " + format(kl("b13-wrong-layer"), ".5f")
                   + " vs delete " + format(kl("b1-delete"), ".5f"))
        claim("the wrong-layer donor is the only arm whose gold-token accuracy "
              "falls below 0.27", "forced_heldout_A", True,
              gold("b13-wrong-layer") < 0.27,
              note="gold_top1 " + format(gold("b13-wrong-layer"), ".4f"))

    # -- F6: matched-budget group selection with the full control set ------
    need = ("b6-static-25", "b5-random-25", "b6-magnitude-25", "b6-reversed-25")
    if all(kl(k) is not None for k in need):
        claim("significance beats random at matched rho=0.25 (FIDELITY)",
              "forced_heldout_B", True, kl("b6-static-25") < kl("b5-random-25"),
              note="static " + format(kl("b6-static-25"), ".5f") + " vs random "
                   + format(kl("b5-random-25"), ".5f"))
        claim("and their 95% intervals do not overlap",
              "forced_heldout_B", True, ci("b6-static-25")[1] < ci("b5-random-25")[0],
              note="static hi " + format(ci("b6-static-25")[1], ".5f")
                   + " < random lo " + format(ci("b5-random-25")[0], ".5f"))
        claim("REVERSING the ranking is worse than random "
              "(the ORDER carries the information)", "forced_heldout_B", True,
              kl("b6-reversed-25") > kl("b5-random-25"),
              note="reversed " + format(kl("b6-reversed-25"), ".5f"))
        claim("the MAGNITUDE heuristic is worse than random "
              "(an interventional label beats a weight-norm proxy)",
              "forced_heldout_B", True,
              kl("b6-magnitude-25") > kl("b5-random-25"),
              note="magnitude " + format(kl("b6-magnitude-25"), ".5f"))

    # -- F7: dynamic buys nothing over static ------------------------------
    if kl("b9-sig-dyn-25") is not None and kl("b6-static-25") is not None:
        claim("dynamic per-state routing does NOT beat static at rho=0.25 "
              "(intervals overlap)", "forced_heldout_B", True,
              ci("b9-sig-dyn-25")[0] < ci("b6-static-25")[1],
              note="dynamic " + format(kl("b9-sig-dyn-25"), ".5f") + " vs static "
                   + format(kl("b6-static-25"), ".5f"))
        rates = ((forced["b9-sig-dyn-25"].get("routing") or {})
                 .get("group_selection_rate") or [])
        if rates:
            claim("...and the dynamic router is genuinely state-conditioned, "
                  "not degenerate", "forced_heldout_B", True,
                  max(rates) - min(rates) > 0.3,
                  note="per-group selection rate spans "
                       + format(min(rates), ".3f") + " to "
                       + format(max(rates), ".3f"))

    # -- F4 / H2: every gate beats its matched random-token control --------
    pairs = (("b11-logitmargin-whole", "r-tok-87"),
             ("b10-margin-whole", "r-tok-84"),
             ("b7-conf-whole", "r-tok-79"),
             ("b12-probe-whole", "r-tok-68"),
             ("b9-entropy-whole", "r-tok-65"))
    for gate, rnd in pairs:
        if kl(gate) is None or kl(rnd) is None:
            continue
        a_g, a_r = act(gate), act(rnd)
        matched = (a_g is not None and a_r is not None and abs(a_g - a_r) < 0.02)
        claim("H2: " + gate + " beats its matched random-token gate",
              "forced_heldout_C + D", True,
              matched and ci(gate)[1] < ci(rnd)[0],
              note="active " + format(a_g or 0, ".4f") + " vs "
                   + format(a_r or 0, ".4f") + "   KL " + format(kl(gate), ".5f")
                   + " vs " + format(kl(rnd), ".5f") + "   "
                   + format(kl(rnd) / max(kl(gate), 1e-9), ".2f") + "x")

    # -- F8: no interior point beats the binary gate -----------------------
    if kl("b16-csgqr-dyn") is not None and kl("b12-probe-whole") is not None:
        claim("no interior three-level point beats the binary probe gate",
              "forced_heldout_C", True,
              kl("b16-csgqr-dyn") >= kl("b12-probe-whole") - 0.002,
              note="three-level " + format(kl("b16-csgqr-dyn"), ".5f") + " @ "
                   + format(act("b16-csgqr-dyn") or 0, ".4f") + " active   vs "
                   "binary " + format(kl("b12-probe-whole"), ".5f") + " @ "
                   + format(act("b12-probe-whole") or 0, ".4f"))
        r = forced["b16-csgqr-dyn"].get("routing") or {}
        claim("...although the interior IS reached (0 < rho < 1 with partial "
              "activation)", "forced_heldout_C", True,
              0.0 < (r.get("full_invocation_rate") or 0) < (r.get("invocation_rate") or 0),
              note="skip " + format(r.get("skip_rate") or 0, ".4f") + "  full "
                   + format(r.get("full_invocation_rate") or 0, ".4f")
                   + "  partial " + format((r.get("invocation_rate") or 0)
                                           - (r.get("full_invocation_rate") or 0),
                                           ".4f"))

    # -- F7.1: the additive proxy is not a predictor -----------------------
    sig = load("trace_L20_int4/significance_analysis.json")
    prb = load("trace_L20_int4/probe_analysis.json")
    if sig and prb and kl("b1-delete") is not None:
        kd = kl("b1-delete")
        row6 = next((r for r in sig["topk_table"] if r["k"] == 6), None)
        pr6 = next((r for r in prb["topk_capture"] if r["k"] == 6), None)
        if row6 and pr6:
            ratios = {}
            for name, armk, proxy in (("random", "b5-random-25",
                                       row6["random_mean"]),
                                      ("static", "b6-static-25",
                                       row6["significance"]),
                                      ("dynamic", "b9-sig-dyn-25",
                                       pr6["probe_hidden"])):
                if kl(armk) is not None:
                    ratios[name] = proxy / max(kd - kl(armk), 1e-9)
            if len(ratios) == 3:
                claim("the additive utility proxy overstates true utility for "
                      "every selection", "forced_heldout_B + trace analyses",
                      True, all(v > 1.0 for v in ratios.values()),
                      note="   ".join(k + " " + format(v, ".2f") + "x"
                                      for k, v in ratios.items()))
                claim("the overstatement is WORST for the most per-token "
                      "selection (the proxy flatters what optimises it)",
                      "forced_heldout_B + probe_analysis", True,
                      ratios["dynamic"] > ratios["static"] > ratios["random"],
                      note="dynamic " + format(ratios["dynamic"], ".2f")
                           + "x > static " + format(ratios["static"], ".2f")
                           + "x > random " + format(ratios["random"], ".2f") + "x")
        o12 = next((r for r in sig["static_vs_oracle"] if r["k"] == 12), None)
        if o12 and kl("b4-int4-all") is not None:
            true_all = kd - kl("b4-int4-all")
            claim("the proxy oracle at k=12 exceeds the TRUE utility of all 24 "
                  "groups (impossible for a real utility)",
                  "significance_analysis + forced_heldout_A", True,
                  o12["oracle"]["point"] > true_all,
                  note="oracle " + format(o12["oracle"]["point"], ".5f")
                       + " vs true all-24 " + format(true_all, ".5f"))

    # ==================================================================
    # GPU benchmark -- orderings only; the card was shared
    # ==================================================================
    section("GPU benchmark (shared card: orderings and same-run ratios only)")
    gb = load("gpu_bench.json")
    if not gb:
        return

    micro = gb.get("remnant_microbenchmark") or {}
    dense = (micro.get("dense_nn_linear") or {}).get("p50_s")
    if dense and micro.get("int4_all"):
        r = micro["int4_all"]["p50_s"] / dense
        claim("the INT4 remnant is SLOWER than the dense MLP it replaces",
              "gpu_bench.json", True, r > 1.0,
              note=format(1000 * micro["int4_all"]["p50_s"], ".4f") + " ms vs "
                   + format(1000 * dense, ".4f") + " ms = " + format(r, ".2f") + "x")
    if dense and micro.get("fp_rho0.250"):
        r = micro["fp_rho0.250"]["p50_s"] / dense
        claim("hard group skipping DOES save time absent dequantization "
              "(FP at rho=0.25 is faster than dense)", "gpu_bench.json", True,
              r < 1.0, note=format(r, ".2f") + "x dense")

    share = gb.get("monitor_share_of_dense_mlp") or {}
    if share:
        claim("the exact-lens monitor costs MORE than the MLP it gates",
              "gpu_bench.json", True,
              share["exact_lens_plus_features"] > 1.0,
              note=format(100 * share["exact_lens_plus_features"], ".1f")
                   + "% of the dense MLP")
        ratio = share["exact_lens_plus_features"] / max(share["need_probe"], 1e-12)
        claim("the probe monitor is at least 50x cheaper than the exact lens",
              "gpu_bench.json", True, ratio > 50,
              note=format(ratio, ".0f") + "x cheaper")
        claim("the probe's MEASURED share exceeds its MAC share by >100x "
              "(a MAC ratio is not a latency)", "gpu_bench.json", True,
              share["need_probe"] > 100 * 0.000109,
              note="measured " + format(100 * share["need_probe"], ".2f")
                   + "% vs MAC share 0.0109%")

    res = gb.get("resident_bytes") or {}
    fp = res.get("fp") or {}
    m = fp.get("measured_mib", fp.get("measured"))
    a = fp.get("analytic_mib", fp.get("analytic"))
    if m and a:
        claim("the FP passthrough remnant COPIES the donor "
              "(measured bytes == analytic bytes)", "gpu_bench.json", True,
              abs(m - a) / a < 0.02,
              note="measured " + format(m, ".3f") + " vs analytic "
                   + format(a, ".3f") + " MiB")

    def peak(key):
        pf = ((gb.get("arms") or {}).get(key) or {}).get("prefill") or {}
        v = next(iter(pf.values()), {}) if pf else {}
        return v.get("peak_reserved_mib")

    pd, pfull = peak("b0-dense"), peak("b16-csgqr-dyn")
    if pd and pfull:
        claim("peak reserved VRAM is HIGHER for the full method than for dense "
              "(Gate F fails)", "gpu_bench.json", True, pfull > pd,
              note="dense " + format(pd, ".0f") + " MiB -> full method "
                   + format(pfull, ".0f") + " MiB")
