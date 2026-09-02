"""Stage E: per-problem teacher-forced NLL from an already-trained V1 adapter.

Why this exists
---------------
The generated-accuracy endpoint is paired BINARY correctness at a 5.54% discordance
rate, which gives a three-seed minimum detectable effect of 1.39 pp -- larger than
the 1.00 pp effect V1 actually measured
([`endpoint_power.py`](endpoint_power.py)). Every remaining DAST question is about
effects at or below that size, so they are unanswerable on it.

A per-problem continuous endpoint is not throttled that way. And it is nearly free:
all twelve V1 runs saved `adapter/` and `dast_auxiliary.pt`, so this needs FORWARD
PASSES ONLY -- no retraining, no generation, no sampling.

What is computed, per problem
-----------------------------
Gold-token negative log-likelihood over the assistant span, exactly the span the LM
loss was trained on (`labels != -100`, i.e. the full reasoning trace plus the final
answer sentence). Reported as a sum, a per-token mean, and a mean gold-token
probability. Also the NLL of the final k gold tokens for k in {8, 16, 32} as an
ANSWER-REGION PROXY -- the corpus's responses end "... the answer is X.", so the
tail is answer-dominated, but this is a proxy and is labelled as one, not a parsed
answer span.

Standing constraint
-------------------
Generated accuracy remains the capability endpoint. This ranks candidates cheaply;
it never promotes one on its own. It must first reproduce the known V1 ordering --
see `teacher_forced_compare.py`, which enforces that gate.

Usage
-----
    python scripts/teacher_forced_endpoint.py --run runs/g3_s29_state_ordered
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dast.config import load_config  # noqa: E402
from dast.data import load_records  # noqa: E402
from dast.tokenization import IGNORE_INDEX  # noqa: E402
from dast.training import _move_batch, build_dataloaders  # noqa: E402
from gradient_conflict import load_trained, trainer_running  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAIL_WINDOWS = (8, 16, 32)


def accelerator_state() -> tuple[int, int, list[str]]:
    """(free MiB, utilisation %, other compute processes). (-1, -1, []) if unknown.

    `trainer_running()` only catches DAST's own trainers. The accelerator is shared
    with the desktop AND with other Rineya worktrees: a fast-weight-attention job
    from a sibling checkout took 6.5 GiB at 100% utilisation part-way through a
    Stage E pass here, which did not fail -- it just ran ~100x slower and produced a
    partial file. Checking headroom once at startup is not enough on its own, but it
    is what turns that into a refusal instead of a silent crawl.
    """

    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60,
        )
        free, utilisation = (
            int(part.strip()) for part in query.stdout.strip().splitlines()[-1].split(",")
        )
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
        others = [
            line.strip() for line in apps.stdout.strip().splitlines()
            if line.strip() and "python" in line.lower()
        ]
        return free, utilisation, others
    except Exception:  # noqa: BLE001
        return -1, -1, []


def resolve_config(run_dir: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    arm, seed = manifest["arm"], int(manifest["seed"])
    for candidate in (
        ROOT / "configs" / f"g3_s{seed}_{arm}.yaml",
        ROOT / "configs" / f"g2_budget_{arm}.yaml",
    ):
        if candidate.exists():
            return candidate
    raise SystemExit(f"cannot infer a config for arm={arm} seed={seed}")


@torch.no_grad()
def per_problem_nll(model, batch: dict) -> list[dict]:
    """Gold-token NLL per example. Computed one example at a time to bound memory.

    A [2, 511, 151936] float32 logit tensor is 621 MB; per-example it is 310 MB,
    which matters on an 8 GiB accelerator shared with the desktop.
    """

    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        prompt_mask=batch["prompt_mask"],
    )
    logits = output.logits
    labels = batch["labels"]
    rows = []
    for index in range(logits.shape[0]):
        shifted_logits = logits[index, :-1, :]
        shifted_labels = labels[index, 1:]
        keep = shifted_labels != IGNORE_INDEX
        if not bool(keep.any()):
            raise ValueError("an example carries no supervised gold tokens")
        token_nll = F.cross_entropy(
            shifted_logits[keep].float(), shifted_labels[keep], reduction="none"
        )
        total = float(token_nll.sum())
        count = int(token_nll.numel())
        row = {
            "n_gold_tokens": count,
            "nll_sum": total,
            "nll_mean": total / count,
            "mean_gold_prob": float(torch.exp(-token_nll).mean()),
        }
        for window in TAIL_WINDOWS:
            tail = token_nll[-window:] if count >= window else token_nll
            row[f"nll_mean_last{window}"] = float(tail.mean())
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-free-mib", type=int, default=3000)
    parser.add_argument("--max-busy-util", type=int, default=60)
    args = parser.parse_args()

    if trainer_running():
        print("REFUSE: a DAST trainer is running; Stage E deferred (GPU exclusivity).")
        return 2
    free, utilisation, others = accelerator_state()
    if free >= 0 and (free < args.min_free_mib or utilisation > args.max_busy_util):
        print(
            f"REFUSE: accelerator busy -- {free} MiB free, {utilisation}% utilised "
            f"(need >={args.min_free_mib} MiB and <={args.max_busy_util}%).\n"
            f"  other compute processes: {others or 'none reported'}\n"
            "  Hardware policy is one GPU job at a time. Competing does not fail, it "
            "just runs ~100x slower and writes a partial file. Re-run when idle."
        )
        return 2
    if free >= 0:
        print(f"accelerator: {free} MiB free, {utilisation}% utilised", flush=True)

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    config = load_config(resolve_config(run_dir, args.config))
    if config.data.eval_split.lower() == "test":
        raise SystemExit("refusing to score the sealed TEST split")

    model, tokenizer = load_trained(run_dir, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    _train_loader, eval_loader, _paths = build_dataloaders(config, tokenizer)
    generators = {
        record.problem_id: record.generator
        for record in load_records(
            config.data_path_for_split(config.data.eval_split),
            split=config.data.eval_split,
            limit=config.data.max_eval_records,
        )
    }

    out_path = Path(args.out) if args.out else run_dir / "teacher_forced_per_problem.jsonl"
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for batch_index, raw in enumerate(eval_loader):
            if args.limit is not None and written >= args.limit:
                break
            batch = _move_batch(raw, device)
            rows = per_problem_nll(model, batch)
            for position, row in enumerate(rows):
                problem_id = raw["problem_ids"][position]
                record = {
                    "problem_id": problem_id,
                    "generator": generators.get(problem_id, ""),
                    "category": raw["categories"][position],
                    "answer_type": raw["answer_types"][position],
                    **row,
                }
                handle.write(json.dumps(record) + "\n")
                written += 1
            if batch_index % 50 == 0:
                print(f"  scored {written}", flush=True)

    mean_nll = 0.0
    with out_path.open(encoding="utf-8") as handle:
        values = [json.loads(line)["nll_mean"] for line in handle]
    mean_nll = sum(values) / len(values)
    print(
        f"\n{run_dir.name}: arm={manifest['arm']} seed={manifest['seed']} "
        f"n={written} mean per-token NLL={mean_nll:.6f}\nwrote {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
