# Remote Deployment Verification

This file is committed directly to the remote branch `research/pre-deletion-recovery-routing-v0.3` as an independent visibility and persistence check.

Verified properties:

- the branch exists on `origin`;
- its merge base is `r2-healing-harness`;
- the PDRR experiment tree is present under `experiments/predeletion_recovery/`;
- the path-scoped CI workflow is present under `.github/workflows/`;
- the branch is isolated from `r2-healing-harness`, `main`, and the other research branches;
- the committed CPU reference is explicitly a mechanism/contract validation, not a real-Rineya quality claim.

Local checkout:

```bash
git fetch origin
git switch --create research/pre-deletion-recovery-routing-v0.3 --track origin/research/pre-deletion-recovery-routing-v0.3
```

For an existing local branch:

```bash
git switch research/pre-deletion-recovery-routing-v0.3
git pull --ff-only
```
