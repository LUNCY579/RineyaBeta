"""Tolerance-free schedule equivalence and registered-topology pins.

A single full forward and a chunked streaming forward necessarily issue
different-width matmuls, so at 28 layers their logits differ by a few ulps.
No absolute logit tolerance can separate that rounding from a genuine cache,
mask or absolute-position fault. These tests therefore compare the ATTENDED
ABSOLUTE KEY POSITIONS, which is exact, and they drive the production
blocked_attention/CompactKV rather than re-deriving the rule (a re-derivation
would agree with a bug in the same way the implementation does).
"""
import unittest
import torch
from ..attention import CompactKV, allowed_mask, blocked_attention
from ..config import ARMS, arm_spec

torch.set_num_threads(1)

WINDOW_BOUNDARIES = (511, 512, 513, 1023, 1024, 1025)


def _tensors(n, seed=17):
    g = torch.Generator().manual_seed(seed)
    shape = (1, 1, n, 2)  # Positions are the object under test, not values.
    return (torch.randn(*shape, generator=g), torch.randn(*shape, generator=g),
            torch.randn(*shape, generator=g))


def full_schedule(n, window, sinks, block):
    q, k, v = _tensors(n)
    pos = torch.arange(n)
    trace = []
    with torch.inference_mode():
        blocked_attention(q, k, v, pos, pos, window=window, sinks=sinks,
                          block_size=block, trace=trace)
    return dict(trace)


def streaming_schedule(n, window, sinks, block, chunks):
    """Drive the real CompactKV merge/attend/evict cycle."""
    q, k, v = _tensors(n)
    pos = torch.arange(n)
    cache, trace = CompactKV(window, sinks), []
    with torch.inference_mode():
        for lo, hi in chunks:
            keys, values, kp = cache.merge(k[:, :, lo:hi], v[:, :, lo:hi], pos[lo:hi])
            blocked_attention(q[:, :, lo:hi], keys, values, pos[lo:hi], kp,
                              window=window, sinks=sinks, block_size=block, trace=trace)
            cache.retain(keys, values, kp)
    return dict(trace), cache


def ideal_schedule(n, window, sinks):
    """Ground truth read straight off the mask definition, with no blocking."""
    pos = torch.arange(n)
    mask = allowed_mask(pos, pos, window, sinks)
    keys = pos.tolist()
    return {q: tuple(key for key, on in zip(keys, mask[row].tolist()) if on)
            for row, q in enumerate(keys)}


def prefill_then_decode(n, block, decode=3):
    prompt = n - decode
    chunks = [(s, min(s + block, prompt)) for s in range(0, prompt, block)]
    return chunks + [(prompt + i, prompt + i + 1) for i in range(decode)]


class ScheduleEquivalenceTests(unittest.TestCase):
    def assert_three_way(self, n, window, sinks, block, chunks):
        ideal = ideal_schedule(n, window, sinks)
        full = full_schedule(n, window, sinks, block)
        stream, _ = streaming_schedule(n, window, sinks, block, chunks)
        self.assertEqual(full, ideal, "blocked full forward diverged from the mask rule")
        self.assertEqual(stream, ideal, "streamed forward diverged from the mask rule")

    def test_registered_window_boundaries(self):
        # The lengths the protocol registers, against the real 512/1024 windows.
        for n in WINDOW_BOUNDARIES:
            for window, sinks in ((512, 0), (512, 4), (1024, 0), (1024, 4)):
                with self.subTest(n=n, window=window, sinks=sinks):
                    self.assert_three_way(n, window, sinks, 128,
                                          prefill_then_decode(n, 128))

    def test_small_exhaustive_grid(self):
        # Block sizes that do and do not divide the length, and that exceed the
        # window; windows of 1; and sinks filling all but one recent slot.
        for n in (1, 2, 7, 8, 9, 16, 17, 33):
            for window in (1, 2, 3, 7, 8, 16, 64):
                for sinks in {0, 1, min(4, window - 1), window - 1}:
                    if not 0 <= sinks < window:
                        continue
                    for block in (1, 3, 7, 8, 128):
                        with self.subTest(n=n, window=window, sinks=sinks, block=block):
                            chunks = ([(s, min(s + block, n)) for s in range(0, n, block)]
                                      if n <= 3 else prefill_then_decode(n, block, 2))
                            self.assert_three_way(n, window, sinks, block, chunks)

    def test_global_layers_are_unaffected_by_chunking(self):
        for n in (513, 1025):
            with self.subTest(n=n):
                self.assert_three_way(n, None, 0, 128, prefill_then_decode(n, 128))

    def test_absolute_positions_survive_eviction(self):
        # After many evictions the retained positions must still be the true
        # absolute ones; renumbering would leave this arithmetic intact but
        # silently move the RoPE phase of every cached key.
        n, window, sinks = 300, 32, 4
        _, cache = streaming_schedule(n, window, sinks, 16, prefill_then_decode(n, 16, 5))
        retained = cache.positions.tolist()
        self.assertEqual(cache.seen, n)
        self.assertEqual(retained[:sinks], list(range(sinks)))
        self.assertEqual(retained[sinks:], list(range(n - (window - sinks), n)))
        self.assertEqual(len(retained), window)
        self.assertEqual(cache.k.shape[2], window)

    def test_retention_carries_one_slot_the_next_query_cannot_reach(self):
        # retain() keeps `window` slots measured from the token just processed,
        # but the following query may only reach back `window - 1` of them, so
        # one resident slot is already unreachable the moment it is stored.
        # Harmless -- Spec.cache_bytes models `window` too -- but pinned here so
        # a later "optimisation" cannot desynchronise storage from accounting.
        n, window, sinks = 64, 16, 0
        trace, cache = streaming_schedule(n, window, sinks, 8, prefill_then_decode(n, 8, 4))
        resident = cache.positions.tolist()
        self.assertEqual(len(resident), window)
        self.assertIn(resident[0], trace[n - 1])  # the query that just ran used it
        following = ideal_schedule(n + 1, window, sinks)[n]  # what query n may reach
        self.assertNotIn(resident[0], following)
        self.assertEqual(len(following), window)  # yet it still sees a full window


class RegisteredTopologyTests(unittest.TestCase):
    """Pin the topology the protocol registers, so no refactor can drift it."""

    def test_r2_periodic_and_frontloaded_placements(self):
        periodic = (3, 7, 11, 15, 19, 23, 27)
        for arm in ("A1_local1024", "A2_local512", "A3_sink512",
                    "A5_local512_gate", "A6_sink512_gate", "A7_sink1024"):
            with self.subTest(arm=arm):
                self.assertEqual(arm_spec(arm, 28).global_layers, periodic)
        self.assertEqual(arm_spec("A9_frontloaded512", 28).global_layers,
                         (0, 1, 2, 3, 4, 5, 27))
        self.assertEqual(arm_spec("A8_all_local512", 28).global_layers, ())
        self.assertEqual(arm_spec("A0_global", 28).global_layers, tuple(range(28)))
        self.assertEqual(arm_spec("A4_global_gate", 28).global_layers, tuple(range(28)))

    def test_placement_control_is_work_matched_at_every_depth(self):
        for layers in list(range(1, 14)) + [19, 26, 27, 28, 29, 30, 36]:
            with self.subTest(layers=layers):
                periodic = arm_spec("A2_local512", layers).global_layers
                front = arm_spec("A9_frontloaded512", layers).global_layers
                self.assertEqual(len(front), len(periodic))
                self.assertEqual(front[-1], layers - 1)
                self.assertEqual(periodic[-1], layers - 1)
                for placement in (periodic, front):
                    self.assertEqual(placement, tuple(sorted(set(placement))))
                    self.assertTrue(all(0 <= i < layers for i in placement))

    def test_placement_control_degenerates_below_five_layers(self):
        # Fewer than five layers leaves no room to move a global earlier, so the
        # control collapses onto its reference. Recorded, not silently relied on.
        for layers in (1, 2, 3, 4):
            with self.subTest(layers=layers):
                self.assertEqual(arm_spec("A9_frontloaded512", layers).global_layers,
                                 arm_spec("A2_local512", layers).global_layers)
        self.assertNotEqual(arm_spec("A9_frontloaded512", 5).global_layers,
                            arm_spec("A2_local512", 5).global_layers)

    def test_every_arm_keeps_a_final_aggregation_layer(self):
        for arm in ARMS:
            for layers in (1, 5, 12, 19, 26, 27, 28, 29, 36):
                with self.subTest(arm=arm, layers=layers):
                    placement = arm_spec(arm, layers).global_layers
                    if arm == "A8_all_local512":
                        self.assertEqual(placement, ())
                    else:
                        self.assertEqual(placement[-1], layers - 1)


if __name__ == "__main__":
    unittest.main()
