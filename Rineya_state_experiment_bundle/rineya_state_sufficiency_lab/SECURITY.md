# Security

## Scope

This is a research harness. It executes no untrusted code, opens no network
listeners and has no authentication surface. The realistic risks are about
*inputs* and *artifacts*.

## Untrusted inputs

**Dataset archives.** `rineya_lab.data.bundle` treats any ZIP as untrusted and
refuses:

- members with absolute paths or `..` traversal (zip-slip);
- symlink members, which can redirect a later write outside the destination;
- archives above the member-count or total-uncompressed-size guards;
- members whose compression ratio exceeds 500:1 (bomb-shaped).

Extraction re-checks containment after path joining. Integrity is verified
against `sha256sums.txt`; a mismatch raises rather than warns.

**Dataset rows** are parsed as JSON only — never `eval`, never `pickle`.

**Model responses** are treated as data. The parser uses `json.loads` and
regular expressions; the conservative repair step never executes anything. A
response that instructs the harness to do something is scored, not obeyed.

## Things that *are* code execution

- `model.trust_remote_code: true` executes code from the model repository.
  It defaults to `false`. Enable it only for checkpoints you trust, and pin
  `model.revision`.
- Torch checkpoint loading in resume (`torch.load`) uses
  `weights_only=False` because optimizer state contains non-tensor objects. Only
  resume from checkpoints you produced. Do not point `training.resume_from` at an
  untrusted directory.
- `configs/*.yaml` are loaded with `yaml.safe_load`, so YAML cannot construct
  arbitrary Python objects.

## Artifacts

Run directories contain raw model generations. If you evaluate on data that is
sensitive or licence-restricted, treat `runs/` accordingly — it is gitignored by
default. `evaluation.save_raw_generations: false` reduces what is written, at the
cost of making manual audit impossible.

Manifests record the command line, environment variables of interest, hostname
and git state. Review before publishing a run directory.

## Reporting a vulnerability

Open an issue describing the impact and a reproduction. For anything you believe
should not be public first, contact the maintainers privately before filing.

Please do not report as vulnerabilities: the reference backend producing
"incorrect" answers (it is a test fixture that does so deliberately), or the
placeholder non-inferiority margin (it is intentionally unfrozen and warns).
