# Pre-Deletion Recovery Routing (PDRR)

PDRR is a preregistered structured-pruning experiment for the Rineya family. It asks whether a component can transfer its functional responsibilities to surviving components **before** it is permanently removed.

The experiment is intentionally stricter than ordinary prune-then-heal training:

```text
intact checkpoint
    -> diagnose the future deletion
    -> notice period while the doomed component remains observable
    -> explicit responsibility transfer into designated survivors
    -> progressive withdrawal of the doomed component
    -> compact-only consolidation with all temporary routing disabled
    -> physical deletion and numerical-equivalence verification
```

The deployed checkpoint contains neither the deleted component nor a Recovery Matrix. The final parameter count must equal ordinary hard pruning of the same structure.

## Central object

For a residual block `d`,

\[
h_{d+1}=h_d+F_d(h_d),
\]

the missing functional contribution under exact identity ablation is

\[
m_d(x,p)=F_d(h_d)=h_{d+1}-h_d.
\]

For surviving recipient `s`, define the newly acquired residual contribution

\[
\Delta r_s(x,p)=r_s^{\mathrm{recovered}}(x,p)-r_s^{\mathrm{anchor}}(x,p).
\]

A temporary responsibility router produces

\[
R_{d,s}(x,p), \qquad \sum_s R_{d,s}(x,p)=1,
\]

and the component-specific objective asks the survivors to reconstruct the missing job:

\[
\mathcal L_{\mathrm{recover}}=
\frac{\sum_{x,p} S_d(x,p)
\,D\!\left(m_d(x,p),\sum_s R_{d,s}(x,p)\Delta r_s(x,p)\right)}
{\sum_{x,p} S_d(x,p)+\epsilon}.
\]

`S_d(x,p)` is an interventional significance signal derived from the intact-versus-ablated teacher distribution. The router is used only to assign loss responsibility. It is never inserted into the deployment forward path.

## Mandatory comparisons

The arm registry includes:

1. intact model;
2. immediate hard pruning;
3. hard pruning + SFT;
4. hard pruning + KD;
5. gradual deletion + KD;
6. gradual deletion + feature KD;
7. Recovery Matrix + KD but **no component-specific recovery loss**;
8. component-specific recovery loss + **random Recovery Matrix**;
9. uniform routing;
10. fixed heuristic routing;
11. learned state-only routing;
12. learned state-position routing without causal significance;
13. full causal state-position recovery routing.

The full method cannot support a special claim unless it beats the relevant controls under paired, compute-matched comparisons.

## Repository layout

```text
configs/                frozen experiment definitions
scripts/                command-line entry points
src/pdrr/               reusable implementation
results/cpu_reference/  committed controlled reference study
PREREGISTRATION.md      frozen hypotheses and claim gates
RUNBOOK.md               CPU/GPU execution instructions
STATUS.md                exact evidence boundary
```

## Fast local validation

From the repository root:

```bash
python -m pip install -r experiments/predeletion_recovery/requirements-cpu.txt
export PYTHONPATH="$PWD/experiments/predeletion_recovery/src"
python -m pytest -q experiments/predeletion_recovery/tests
python experiments/predeletion_recovery/scripts/run_cpu_reference.py --quick
```

On PowerShell:

```powershell
python -m pip install -r experiments/predeletion_recovery/requirements-cpu.txt
$env:PYTHONPATH = "$PWD\experiments\predeletion_recovery\src"
python -m pytest -q experiments/predeletion_recovery/tests
python experiments/predeletion_recovery/scripts/run_cpu_reference.py --quick
```

## Real Rineya execution

The real-model scripts require the local ignored checkpoint weights and evaluation sets. See `RUNBOOK.md`. The committed CPU reference is a controlled mechanism test, not a claim about a trained Rineya checkpoint.
