"""Exploratory paired world-cluster analysis; no confirmation or seed pooling."""
from collections import defaultdict
import random
from .identity import ContractError, positive_int


def compare(reference, candidate, seed=17, draws=2000):
    positive_int(draws, "draws", 100000)
    def index(rows):
        out = {}
        for row in rows:
            if row["case_id"] in out or type(row["semantic_correct"]) is not bool:
                raise ContractError("Duplicate case or non-boolean outcome")
            out[row["case_id"]] = row
        return out
    a, b = index(reference), index(candidate)
    if not a or a.keys() != b.keys():
        raise ContractError("Paired cases must match exactly")
    groups, families = defaultdict(list), defaultdict(list)
    wins = losses = 0
    for key in sorted(a):
        x, y = a[key], b[key]
        for field in ("cluster_id", "op", "renderer", "bank_identity", "model_identity"):
            if x[field] != y[field]:
                raise ContractError(f"Unmatched {field}")
        delta = int(y["semantic_correct"]) - int(x["semantic_correct"])
        groups[x["cluster_id"]].append(delta)
        families[x["op"]].append(delta)
        wins += delta > 0
        losses += delta < 0
    means = [sum(v) / len(v) for v in groups.values()]
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(draws))
    result = {"evidence_class": "EXPLORATORY_WORLD_CLUSTER_ONLY", "worlds": len(means),
              "rows": len(a), "mean_world_delta": sum(means) / len(means),
              "bootstrap_percentile95": [samples[int(.025 * (draws - 1))], samples[int(.975 * (draws - 1))]],
              "row_wins": wins, "row_losses": losses,
              "family_deltas": {k: sum(v) / len(v) for k, v in families.items()},
              "seed": seed, "draws": draws,
              "warning": "Empirical interval; no zero-harm, multiplicity, equivalence or population-capacity conclusion"}
    if all(len(v) == 1 for v in groups.values()):
        from math import comb
        n = wins + losses
        result["exact_mcnemar_two_sided"] = min(1., 2 * sum(comb(n, j) for j in range(min(wins, losses) + 1)) / 2**n) if n else 1.
    return result
