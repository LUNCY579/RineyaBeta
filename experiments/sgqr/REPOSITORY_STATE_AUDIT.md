# SGQR Repository State Audit

## Audit date

2026-08-24

## Repository

`LUNCY579/RineyaBeta`

## New branch

`research/sgqr-core-quantized-remnant`

## Base branch

`r2-healing-harness`

## Starting base commit

`1664f54de0bbdfde45e311b75bbf3038dbfffe42`

## Initial state

The new SGQR branch was created directly from `r2-healing-harness`.

At branch creation:

- ahead of base: 0 commits;
- behind base: 0 commits;
- status: identical;
- CSGQR files: none.

The pre-existing branch `research/csgqr-quantized-remnant` was not used as the
implementation base because it had previously been verified as identical to the
same baseline and contained no substantive CSGQR implementation.

## Isolation decision

SGQR is intentionally isolated from CSGQR.

- **SGQR** fixes the active group budget and evaluates quantized donor
  preservation plus significance routing.
- **CSGQR** may later add confidence as a separate budget controller.

No file on `r2-healing-harness` or the existing CSGQR branch is modified by this
research track.

## Required final verification

Before treating this branch as implemented, verify remotely:

```bash
git fetch origin
git rev-parse origin/research/sgqr-core-quantized-remnant
git rev-list --left-right --count \
  origin/r2-healing-harness...origin/research/sgqr-core-quantized-remnant
git diff --stat \
  origin/r2-healing-harness...origin/research/sgqr-core-quantized-remnant
```

The branch must be ahead of the base and the expected SGQR files must be readable
from the remote branch.
