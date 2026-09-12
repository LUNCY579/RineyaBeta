"""CPU mechanical checks that must pass before inherited frozen DEV is opened."""
import json
from pathlib import Path

import torch

from .interventions import (SinkIntervention, apply_sink_intervention,
                            attention_diagnostics, persistent_mask,
                            select_persistent_positions)


def run_checks():
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    arms = protocol["arms"]
    assert len(arms) == 11 and len({a["id"] for a in arms}) == 11
    assert any(a["count"] == 4 and a["visible"] is False for a in arms)
    assert protocol["execution"]["training_authorized"] is False
    qp = kp = torch.arange(20)
    for count in (0, 1, 2, 4, 8):
        positions = select_persistent_positions(20, count, "prefix")
        mask = persistent_mask(qp, kp, window=12, persistent_positions=positions)
        assert not (mask & (kp[None, :] > qp[:, None])).any()
        assert int(mask[-1].sum()) == 12
    k = torch.arange(1 * 2 * 20 * 3, dtype=torch.float32).reshape(1, 2, 20, 3)
    v = k + 1000
    pos = select_persistent_positions(20, 4, "prefix")
    for mode in ("zero_k", "zero_v", "permute_v", "mean_v"):
        spec = SinkIntervention(4, "prefix", mode)
        ko, vo = apply_sink_intervention(k, v, kp, pos, spec)
        assert torch.equal(ko[:, :, 4:], k[:, :, 4:])
        assert torch.equal(vo[:, :, 4:], v[:, :, 4:])
        if mode == "zero_k":
            assert torch.count_nonzero(ko[:, :, :4]) == 0 and torch.equal(vo, v)
        if mode == "zero_v":
            assert torch.count_nonzero(vo[:, :, :4]) == 0 and torch.equal(ko, k)
    probs = torch.softmax(torch.randn(1, 2, 3, 20), -1)
    metrics = attention_diagnostics(probs, v, range(4))
    assert all(torch.isfinite(x).all() for x in metrics.values())
    return {"status": "PASS", "checks": 7, "arms": len(arms),
            "frozen_dev_unlocked_mechanically": True,
            "note": "checkpoint/source qualification remains separately required"}


if __name__ == "__main__":
    print(json.dumps(run_checks(), sort_keys=True))
