"""New, open synthetic DEV only. Never imports repository evaluation banks."""
import hashlib
import random
import statistics


def graph_rows(role, count, seed):
    if role not in ("train", "dev") or type(count) is not int or count < 1:
        raise ValueError("Only positive-size open train/dev datasets are supported")
    salt = int(hashlib.sha256(f"qref-v1:{role}:{seed}".encode()).hexdigest()[:16], 16)
    rng = random.Random(salt)
    rows = []
    for i in range(count):
        hops = 1 + i % (2 if role == "train" else 4)
        # Namespaces make train/dev entities and individual graphs disjoint.
        names = [f"{'T' if role == 'train' else 'D'}{i:x}_{j:x}" for j in range(16)]
        rng.shuffle(names)
        chain = names[:hops+1]
        edges = list(zip(chain, chain[1:]))
        distractors = names[hops+1:]
        edges += list(zip(distractors[::2], distractors[1::2]))
        rng.shuffle(edges)
        facts = "\n".join(f"{a} links to {b}." for a, b in edges)
        prompt = (f"Facts:\n{facts}\nQuestion: Starting at {chain[0]}, follow exactly "
                  f"{hops} links. Return only the final ID.\nAnswer:")
        rows.append({"id": f"{role}:{seed}:{i}", "hops": hops, "prompt": prompt,
                     "answer": chain[-1], "edges": edges, "start": chain[0]})
    return rows


def paired_summary(records):
    """Training seed is the replication unit. No row-wise pseudoreplication.

    Report seed deltas and a descriptive seed-bootstrap interval; with three
    training seeds this is not a confirmatory significance test.
    """
    arms = ("baseline", "adapter", "independent", "feedback")
    cells = {}
    for r in records:
        key = (r["arm"], r["seed"], r["id"])
        if r["arm"] not in arms or key in cells or type(r["correct"]) is not bool:
            raise ValueError("Invalid/duplicate prediction record")
        cells[key] = r["correct"]
    if not cells:
        raise ValueError("Empty predictions")
    seeds = sorted({s for _, s, _ in cells})
    ids = {i for a, s, i in cells if a == "feedback" and s == seeds[0]}
    if not ids:
        raise ValueError("Missing feedback arm")
    means = {}
    for arm in arms:
        means[arm] = {}
        for seed in seeds:
            observed = {i for a, s, i in cells if a == arm and s == seed}
            if observed != ids:
                raise ValueError("Every arm/seed must score identical DEV rows")
            means[arm][seed] = statistics.mean(cells[arm, seed, i] for i in ids)
    comparisons = {}
    rng = random.Random(90817)
    for control in ("baseline", "adapter", "independent"):
        deltas = [means["feedback"][s] - means[control][s] for s in seeds]
        interval = None
        if len(seeds) >= 3:
            boot = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
            interval = [boot[49], boot[1949]]
        comparisons[control] = {"seed_deltas": dict(zip(seeds, deltas)),
                                "mean_delta": statistics.mean(deltas),
                                "descriptive_seed_bootstrap_95": interval}
    return {"accuracy_by_arm_seed": means, "feedback_vs": comparisons,
            "replication_unit": "training_seed", "confirmatory": False}
