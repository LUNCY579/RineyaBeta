# HLGA-1: first real-R2 qualification attempt

Branch `agent/hlga-continuation-20260908`, continued from
`origin/research/hybrid-local-global-attention-v1` at
`a196318ec75cacccc8ac9ec446e4c4b6a29c2802`.

**Highest stage reached: P2 attempted and FAILED under the registered
instrument. P3 was not opened.** The mechanism qualified on every exact
criterion; the failure is in the qualification instrument, not the retrofit.

Environment: Python 3.11.15, torch 2.6.0+cu124, transformers 5.14.1,
safetensors 0.8.0, tokenizers 0.22.2, Windows 11 (10.0.26200),
AMD Ryzen, NVIDIA GeForce RTX 4070 Laptop GPU (8187 MiB),
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, float32, one thread.

---

## P1 -- real R2 asset identity: ESTABLISHED

The branch tracks `artifacts/rineya-core-en32k-28L` as config plus tokenizer
only; `*.safetensors` is git-ignored, so no worktree carries weights. A
filesystem sweep of 933 `config.json` directories (819 of them Qwen3) found
exactly three checkpoints matching the registered R2 shape *and* holding real
weights. All three carry a byte-identical `model.safetensors`:

    sha256 a6bc04ca357d446d092789c998ddae658a6e405946f6449ae68be28dbe10c562
    948,078,864 bytes

The three differ only in tokenizer packaging, and split into two families:
the two `config.json` variants are byte-different but semantically identical
(formatting only), while `tokenizer.json` and `merges.txt` genuinely differ.
The worktree's branch-tracked assets are the canonical family, so the weights
were copied in beside them (a copy, not a link; nothing outside this worktree
was modified) and re-hashed after the copy.

`checkpoint_id = 409eb848c56983377c5d9d975b1deffdbfced992aaee35b349a1a36ffadb6fab`

| file | bytes | sha256 |
|---|---:|---|
| model.safetensors | 948,078,864 | a6bc04ca357d446d092789c998ddae658a6e405946f6449ae68be28dbe10c562 |
| config.json | 724 | 1561e79bbacc1c9f476cfa1242ba423f63d401d84289d8c8bafad03ec14efad8 |
| generation_config.json | 210 | 5584f2c7ddbfbfb37f9d4ab2c9b3dc2e5f22c2911dc89401817e6ed03ee17d26 |
| tokenizer.json | 2,268,138 | 4b37e7762462cd5695283234320f69898ad3bb3e9cf4b191757d8e6e7181063d |
| tokenizer_config.json | 9,706 | e468e9c71bd427453cbd0872d722c50f069b36c71a13d4bc0e49fd4bf1fb0593 |
| merges.txt | 275,128 | 3c9e86f5961936d99fe8dd4cf060c0bc525d2f6de0357a084acba70853834dc4 |
| vocab.json | 562,397 | f3bfe6ff4d099be980f1306024cf1e89f9061febc91c3a4d312e185c221c1537 |

`assert_target_config` passes on the raw config: qwen3, 28L, hidden 1024,
16Q/8KV, head_dim 128, MLP 3072, vocab 32768, max_pos 40960, theta 1e6,
tied embeddings, stored dtype bfloat16 (loaded and verified as float32).

Shape and hashes fix *which bytes* were used. They are not lineage proof:
nothing local attests that these weights descend from the Rineya R2 training
run. `trim_manifest.json` records a derivation from Qwen3-0.6B (32,742 base +
26 control tokens, 32,304 merges kept, 119,083 dropped, 182 unreachable vocab
rows), which is consistent with R2's documented construction but is a
self-declaration, not an independent attestation.

---

## P2 -- mechanical qualification: FAILED, and the instrument is why

Four runs on one code identity `eb2dd6c5b1ff`, all
`checkpoint_unchanged: true`, `source_unchanged: true`. `max_abs_error`,
P = passed / F = failed against the registered `atol = rtol = 2e-05`:

| check | cpu/math | cpu/sdpa | cuda/math | cuda/sdpa |
|---|---|---|---|---|
| native_global | 0.000e+00 P | 1.526e-05 P | 0.000e+00 P | 2.766e-05 **F** |
| gate_identity | 0.000e+00 P | 0.000e+00 P | 0.000e+00 P | 0.000e+00 P |
| global_stream | 2.003e-05 P | 2.289e-05 P | 3.624e-05 P | 3.147e-05 P |
| sink_stream | 1.812e-05 P | 2.289e-05 P | 3.624e-05 P | 3.147e-05 P |
| local_stream | 9.346e-05 **F** | 7.749e-05 **F** | 1.106e-04 **F** | 6.485e-05 **F** |
| all_local_stream | 9.203e-05 **F** | 1.106e-04 **F** | 1.822e-04 **F** | 9.680e-05 **F** |
| frontloaded_stream | 1.221e-04 **F** | 7.200e-05 **F** | 7.534e-05 **F** | 1.116e-04 **F** |

In every run, in every arm: `attended_positions_identical: true`,
`first_divergent_layer: null`, `argmax_identical: true`,
`compact_storage: true`.

### What passed exactly

* **Native parity is bit-exact.** The research A0_global forward reproduces
  the native HF Qwen3 eager forward with `max_abs_error` exactly `0.0` on both
  math-backend runs. The independent oracle and the research path agree to the
  last bit.
* **Identity gates are bit-exact.** `2*sigmoid(0) = 1.0` exactly in float32,
  so A0 == A4 with `max_abs_error 0.0` in all four runs.
* **Mask, cache, eviction and absolute positions are exactly correct.**
  Instrumenting the real 28-layer forward, the set of absolute key positions
  each query attends to is identical between the full and chunked schedules in
  every arm: 0 of 28 layers diverge, with no tolerance involved anywhere.
* **Storage is compact** (`logical_kv_bytes == resident_kv_bytes`) and the
  per-layer retained counts match `Spec.cache_bytes`.

### Why the tolerance failed

A single full forward and a chunked streaming forward are mathematically
identical but *numerically* distinct: chunking the query dimension changes the
gather width, so the same softmax is accumulated over differently shaped
matmuls. Isolating layer 0 of the real model, with the softmax allowed to
follow the input dtype, full-vs-stream differs by

    float32: 1.2368e-06  =  11.8 ulps
    float64: 2.9698e-15  =  15.2 ulps

Error that tracks machine epsilon is non-associativity, not a cache or mask
fault. (Left at the shipped `softmax(dtype=torch.float32)`, the float64 case
floors at 1.06e-07 -- almost exactly one *float32* ulp -- which is the pinned
softmax precision showing through, not a semantic residue.) Over 28 residual
layers this compounds to roughly 1e-4 in the logits.

This is not a windowing effect. A0_global drifts the same way, and its
`global_stream` check cleared 2e-5 only through `rtol`, at 2.0027e-05 on
cpu/math. The pass/fail split tracks rounding luck rather than mechanism:
`sink_stream` clears the bar at 3.624e-05 on cuda/math while `local_stream`
fails at 1.106e-04, and the two arms differ only by four sink slots. On
cuda/sdpa even `native_global`, the independent parity oracle, fails.

**An absolute logit tolerance of 2e-05 was calibrated on the 4-layer,
hidden-32 CPU fixture and cannot be met at R2 scale by any correct
implementation.** Every prior receipt on this branch carries
`real_rineya_checkpoint_loaded: false`, so this is the first time the
registered instrument met the real 28-layer model, which is why the
mis-calibration had not surfaced.

The threshold was **not** changed. The registered comparison stands exactly as
written and the receipts record `passed: false`. What was added is strictly
additive: an exact, tolerance-free comparison of attended absolute key
positions, bound into the existing checks with AND so that it can only ever
withhold a pass.

---

## Defects found and fixed

1. **The branch head did not pass its own test suite** (3 errors of 99).
   * `validate_model` rejected every real R2 checkpoint. Transformers 4.x
     recorded "no RoPE scaling" as `rope_scaling = None`; Transformers >= 5
     materializes `{"rope_type": "default", "rope_theta": T}` and removes
     `rope_theta` from the config root. Both encode the same unscaled RoPE, so
     `rope_scaling is not None` read an unmodified checkpoint as scaled.
     **This alone blocked P2 entirely.**
   * `score()` raised a bare `KeyError` on any case lacking the schema-v2
     attribution fields, and the committed test still built a v1 case.
   * `test_symlink_artifact_refused` errored with WinError 1314 (Windows
     denies symlink creation without privilege).
2. **The registered CUDA path could never run.** `main()` enables
   `torch.use_deterministic_algorithms(True)`, which refuses every CuBLAS GEMM
   on CUDA >= 10.2 unless `CUBLAS_WORKSPACE_CONFIG` is set beforehand. The
   registered CUDA constraints had never been exercised.
3. **Qualification had no tolerance-free equivalence check**, so a rounding
   difference and a genuine cache fault were indistinguishable.

Added: `model_fingerprint()` binding the loaded module into receipts; a
load-time assertion that parameters really are float32; `tests/test_boundaries.py`
covering 511/512/513 and 1023/1024/1025 against real 512- and 1024-wide
windows plus an exhaustive small grid, and pinning the registered topology.
Suite: **116/116 pass**, 0 skipped.

---

## P3 instrument analysis (design only -- no model forward, no P3 evidence)

Computed from `make_cases` and the window rule alone. For each registered DEV
case, does a *local* layer still reach the answer tokens? Global layers always
do; the contrast lives exactly where local access is lost.

At W=512 (A2/A3), answer tokens outside the local window:

* at the last prompt token: **12 of 72** cases (lengths 1023/1024/1025, depth 0.0)
* by the end of a 24-token generation: **24 of 72** (adds depth 0.5 at those lengths)

At W=1024 (A1/A7): **0 of 72** at either point. The registered lengths stop at
1025, so a 1024-wide window always retains the answer. **A1_local1024 and
A7_sink1024 cannot exhibit any window effect on this DEV set**; the
`window_budget` and `sink_at_1024` contrasts are uninformative as registered.

The four prefix sinks never hold evidence. The earliest evidence token across
all 72 cases is at position 21, while sinks cover 0..3, which decode to
`'Read the records.'`. The A2-vs-A3 contrast therefore tests whether retaining
the *instruction prefix* and stabilising the attention sink helps -- not
whether sinks preserve evidence.

Consequence for reading P3 when it is licensed: the overall 72-case McNemar is
diluted roughly 3:1 by cells where the mechanism cannot bite, and with only
12-24 informative pairs it has essentially no power (six discordant pairs all
one way gives p = 0.031; five gives 0.063). The stratified
`by_length` x `by_depth` cells at 1023/1024/1025, which `summarize` already
emits, are where the signal must be read.

---

## What this does and does not establish

**Does:** the R2 weights in use are identified by hash; the research runtime
reproduces native HF Qwen3 bit-exactly; identity gates are exactly identity;
the periodic local/global mask, compact KV storage, eviction and absolute
positions are exactly correct on the real 28-layer model across CPU and CUDA
and both backends; the registered topologies are what the protocol claims.

**Does not:** license P3. Say anything about model quality, capability
preservation, damage, speed or memory. Establish weight lineage. Establish
that a local/global retrofit is *scientifically* viable -- only that it is
*mechanically* faithful.

## Recommendation

Continue HLGA, with one amendment first. The single highest-value next step is
a **preregistered replacement for the P2 equivalence criterion**, written and
signed off before P3 is run: keep exact attended-position identity, exact
argmax identity and bit-exact native/gate parity as pass conditions, and
replace the absolute logit `allclose` with a bound derived from a control
rather than a constant -- for example, requiring the local arms' full-vs-stream
drift to be within a small factor of the same drift measured on A0_global,
which is known-correct by bit-exact native parity. That is a *stronger*
instrument, and fixing it is a prerequisite for any P3 result to mean anything.

The registered DEV set should be amended in the same pass, since as it stands
two of the ten arms cannot move.
