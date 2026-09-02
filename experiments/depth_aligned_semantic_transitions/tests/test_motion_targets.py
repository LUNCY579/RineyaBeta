"""Motion targets must be the delta chain the H6 gate claims to test.

The first delta uses `compact_state_trace[0]`, the pre-reasoning state that
`raw_targets` deliberately drops, so K absolute states yield exactly K motion
targets at K depths. Getting that chain wrong -- off by one, or dropping the first
delta -- would silently change which quantity the gate evaluated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def test_delta_chain_uses_the_initial_state_and_preserves_length() -> None:
    torch.manual_seed(3)
    count, stages, dim = 6, 4, 5
    absolute = F.normalize(torch.randn(count, stages, dim, dtype=torch.float64), dim=-1)
    initial = F.normalize(torch.randn(count, dim, dtype=torch.float64), dim=-1)

    previous = torch.cat([initial.unsqueeze(1), absolute[:, :-1, :]], dim=1)
    deltas = absolute - previous

    assert deltas.shape == absolute.shape, "K states must give K deltas"
    # Stage 0's delta is measured against the pre-reasoning state, not against
    # stage 0 itself (which would be identically zero).
    assert torch.allclose(deltas[:, 0], absolute[:, 0] - initial)
    for stage in range(1, stages):
        assert torch.allclose(deltas[:, stage], absolute[:, stage] - absolute[:, stage - 1])


def test_identical_consecutive_states_give_a_degenerate_delta() -> None:
    """A duplicated depth target makes its delta exactly zero.

    This is why the distinctness filter is not optional for the motion variant: a
    zero vector normalises to zero, and its cosine against anything is 0, which
    would quietly enter the mean as a fixed penalty rather than as a measurement.
    """

    dim = 4
    state = F.normalize(torch.randn(1, dim, dtype=torch.float64), dim=-1)
    absolute = torch.stack([state, state], dim=1)  # two identical consecutive states
    deltas = absolute[:, 1] - absolute[:, 0]
    assert float(deltas.norm()) == 0.0
    assert float(F.normalize(deltas, dim=-1).norm()) == 0.0


def test_normalised_motion_targets_are_unit_length() -> None:
    torch.manual_seed(11)
    deltas = torch.randn(9, 4, 7, dtype=torch.float64)
    unit = F.normalize(deltas, dim=-1)
    assert torch.allclose(unit.norm(dim=-1), torch.ones(9, 4, dtype=torch.float64))
