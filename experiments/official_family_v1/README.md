# Official Rineya dense-core experiment migration

This is a **new target contract**, not a rename of R2 and not a performance result.
The two successor branches keep their original experiment implementations and
historical evidence unchanged. New entry points reuse the algorithms but cannot
reuse an R2 receipt. Checkpoint, tokenizer, source, protocol, device, precision or
runtime drift requires fresh qualification.

## Registered checkpoint sources

The authoritative release ledger is `LUNCY579/ChatBubble`, commit
`a6ee824584a9e28881826ab92a11db48e124d90c`, under `Rineya_Published_Cores`.
`profiles.json` pins exact core IDs, weight SHA-256 values, file sizes and available
metadata/tokenizer reference hashes. The ledger explicitly states that weights
are local and gitignored: cloning either GitHub repository is not sufficient.

| Explicit profile | Immutable dense directory | Weight bytes |
|---|---|---:|
| `supa-1.7.0` | `Rineya-Supa-1.7.0-devfreeze` | 1,192,135,096 |
| `nimbus-1.6.0` | `Rineya-Nimbus-1.6.0-devfreeze` | 1,503,300,328 |
| `halo-1.2.0` | `Rineya-Halo-1.2.0-devfreeze` | 2,384,234,968 |

The private backup `Luncy1978/rineya-published-cores` was confirmed accessible
through the owner's connected account during this migration. The documented
backup revision is `b14103c71c240998171694a0d0397bd4e5d64d71`. Its actual binary
contents were **not downloaded or pretrained-qualified during implementation**.
Restore checks that revision's directory listing and hashes before accepting a
core; absence or metadata-version drift is an error, never permission to fetch
another model or rewrite the source files.

Start with **Supa Core 1.7** as an operational choice: its release ledger includes
weight and tokenizer reference hashes and it uses the standard dense runtime.
This is not a claim that Supa wins these new tasks. Nimbus and Halo are independent
replications with separate receipts, not interchangeable fallback names.
Supa Core 1.7 is **not** Supa-A1, Supa-A1.1, or Supa 2.0 System48. The latter names
include custom policies or routed systems. Halo is not Halo Uncensored; Nimbus
1.6 is not the archived Nimbus 1.3 release. These alternatives, GGUF, quantized
exports, adapter-only files, and unregistered shards are rejected.

## Supply or restore exactly one core

Required files in the selected core directory:
`CORE.json`, `config.json`, `model.safetensors`, `tokenizer.json`, and
`tokenizer_config.json`. Preserve available generation config, chat template,
special-token maps, added tokens, vocabulary and merge files. Runtime assets are
fingerprinted; filenames alone are not evidence. Use materialized regular files,
not unresolved Git LFS pointers or HF cache symlinks. Official unsharded bytes are
required; reserialized equivalent weights need a separately reviewed new profile.

In a dedicated environment, install the correct PyTorch build for the machine,
then `python -m pip install -r experiments/official_family_v1/requirements.txt`.
Native execution pins Transformers 4.57.1, eager attention and FP32. Source files
are untouched when Halo's default `rope_parameters` representation is translated
in memory into explicit `rope_theta=1000000` for that runtime. Nondefault/scaled
RoPE is rejected. Older numerical parity is not presumed after this translation.

All commands below run from the repository root. Use a writable external model
and run directory. The restore command is the **only network-enabled path** and
uses the caller's ordinary authenticated Hugging Face credentials. The connected
chat account does not automatically install credentials on another machine.
Never put a token in the command, a committed file or a receipt.

```bash
python -m experiments.official_family_v1 restore \
  --profile supa-1.7.0 --destination ../rineya-models/Supa-1.7 \
  --out ../rineya-runs/supa-restoration.json

# Or inspect an already materialized official release, without network access:
python -m experiments.official_family_v1 inventory \
  --profile supa-1.7.0 --model ../rineya-models/Supa-1.7 \
  --out ../rineya-runs/supa-inventory.json
```

Restore never overwrites an existing directory and copies only selected data
files. It never downloads/executes custom model code or starts optimization.
For Halo, the inspected provenance lacks an independent `tokenizer.json`
reference hash. Its tokenizer bytes are content-bound and its IDs validated,
but `tokenizer_reference_hash_verified` remains **false**. Do not describe that
as the same provenance strength as the Supa/Nimbus tokenizer hash checks.

## Query-refinement successor

Branch: `research/query-refinement-official-family-v1`, based on
`e5794507c4e4236068b3124ce6efe61ed8dc9e56`.
Entry point: `experiments.query_refinement_v1.family_pipeline`.

```bash
python -m experiments.query_refinement_v1.family_pipeline qualify \
  --profile supa-1.7.0 --model ../rineya-models/Supa-1.7 \
  --device cuda:0 --out ../rineya-runs/supa-query-qualify

python -m experiments.query_refinement_v1.family_pipeline smoke \
  --profile supa-1.7.0 --model ../rineya-models/Supa-1.7 \
  --device cuda:0 --receipt ../rineya-runs/supa-query-qualify/receipt.json \
  --out ../rineya-runs/supa-query-smoke

python -m experiments.query_refinement_v1.family_pipeline dev \
  --profile supa-1.7.0 --model ../rineya-models/Supa-1.7 \
  --device cuda:0 --receipt ../rineya-runs/supa-query-smoke/receipt.json \
  --out ../rineya-runs/supa-query-dev
```

Qualification uses the unmodified native forward as its independent oracle;
checks zero-intervention identity, enabled-path reachability, cached/full and
SDPA/eager parity, causality, finite active adapter gradients and frozen backbone.
The same checks run again before smoke and DEV. Layer 14, rank 8, and the four
baseline/adapter/independent/feedback arms are retained. Nonbaseline updates have
2,064 parameters. Smoke is 25 steps with one seed; open DEV uses the three
registered seeds and 100 steps. Training-example, answer-token, optimizer and
step matching remain intact. Baseline is not falsely described as compute-matched
to two-read trained arms. DEV does not authorize sealed evaluation, confirmatory
claims or production promotion. Each profile is its own experimental stratum.

## Composition-QRR successor

Branch: `research/composition-qrr-official-family-v1`, based on
`690dcf27ee48512369fc2f3ca02603dee06ddfba`.
Entry point: `experiments.composition_qrr_v1.family_native`.

```bash
python -m experiments.composition_qrr_v1.family_native prepare \
  --profile supa-1.7.0 --model ../rineya-models/Supa-1.7 \
  --device cuda:0 --out ../rineya-runs/supa-qrr-contract.json
```

Inspect that draft's exact asset, source, runtime, protocol and budget binding.
Only a reviewer may change `status` from `UNREVIEWED` to `REVIEWED_FROZEN_ONLY`
and supply `reviewed_by`. Preparing a draft grants no execution permission.
After review:

```bash
python -m experiments.composition_qrr_v1.family_native qualify \
  --profile supa-1.7.0 --model ../rineya-models/Supa-1.7 \
  --device cuda:0 --contract ../rineya-runs/supa-qrr-contract.json \
  --out ../rineya-runs/supa-qrr-qualify

python -m experiments.composition_qrr_v1.family_native diagnose \
  --profile supa-1.7.0 --model ../rineya-models/Supa-1.7 \
  --device cuda:0 --contract ../rineya-runs/supa-qrr-contract.json \
  --receipt ../rineya-runs/supa-qrr-qualify/receipt.json \
  --bank ../rineya-runs/open-composition-bank.json \
  --out ../rineya-runs/supa-qrr-diagnostics
```

Generate/audit the open composition bank through the inherited instrument CLI
as described in that experiment's original README. This path retains its bank
schema, independent oracle, scoring and diagnostic arms. Limits remain 768 cases,
4,096 prompt tokens and 24 generated tokens. No truncation or silent row dropping.
Cached/full mechanical parity is rerun before diagnostics. Oracle-assisted arms
are not deployable improvements; fixed-order timings are descriptive.
**Native QRR intervention qualification and all QRR optimization stay blocked.**
A different checkpoint cannot substitute for those missing integration gates.

## Runtime, provenance and testing boundaries

These dense cores have 28 layers, width 1,024, 16 query heads, 8 KV heads,
128-dimensional heads, 151,936 embedding rows and EOS 151645. R2's 32,768-row
vocabulary/EOS and 474M count are not reused. The loaded tied dense model must
have exactly 596,049,920 unique parameters. Tokenizer vocabulary IDs must fit the
embedding rows; vocabulary length need not equal a padded embedding matrix.
Loading is local-only with built-in Qwen3/Qwen2 classes, safetensors, no remote
code, and no missing/unexpected/mismatched state tensors accepted.

Use `--device cpu` explicitly for a CPU run; there is no automatic CUDA fallback.
Changing device invalidates preceding receipts. CUDA requires 6,144 MiB free
before loading. This is a preflight floor, **not a memory-fit guarantee**. FP32
backbone weights alone are about 2.22 GiB, with additional activation, logits,
optimizer and runtime memory. CPU runs require sufficient host RAM and may be
slow. BF16 or quantized experiment execution requires a separate numerical
qualification; source storage dtype is not the experiment execution dtype.

Receipts bind complete selected runtime assets, shared and experiment source,
protocol and runtime. Post-run rehashes catch changes before a pass is written.
Predecessor output files are checked against their manifests. Output directories
are exclusive and separate from checkpoint/source directories. Content hashes
are integrity checks, not signatures or proof against a malicious forged receipt;
mechanical checks are rerun rather than trusting a supplied pass flag alone.

The workflow `.github/workflows/official-family.yml` runs shared tests plus the
inherited branch's tests, with native **tiny random-weight Qwen3 fixtures** and
network-disabled model loading. It preserves logs/JUnit/summary artifacts.
Passing CI is mechanical evidence only, not a pretrained family run. No private
weights, credentials or model binaries are committed. Historical R2 evidence is
left intact and cannot be promoted into this experiment's qualification.
