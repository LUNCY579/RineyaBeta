"""Strict, dimension-independent ablations. A window includes the current token."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Spec:
    layers: int
    global_layers: tuple[int, ...]
    window: int = 512
    sinks: int = 0
    gate: bool = False
    block_size: int = 128
    backend: str = "math"

    def __post_init__(self):
        for name in ("layers", "window", "block_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.sinks) is not int or not 0 <= self.sinks < self.window:
            raise ValueError("Require 0 <= sinks < window; sinks consume the window budget")
        if type(self.gate) is not bool:
            raise ValueError("gate must be boolean")
        if (not isinstance(self.global_layers, tuple)
                or any(type(i) is not int for i in self.global_layers)
                or tuple(sorted(set(self.global_layers))) != self.global_layers
                or any(not 0 <= i < self.layers for i in self.global_layers)):
            raise ValueError("global_layers must be a sorted, unique, in-range tuple")
        if self.backend not in ("math", "sdpa"):
            raise ValueError("Only explicit math and sdpa backends are supported")

    def local_window(self, layer):
        if not 0 <= layer < self.layers:
            raise IndexError(layer)
        return None if layer in self.global_layers else self.window

    def to_dict(self):
        return asdict(self)

    def cache_bytes(self, tokens, kv_heads, head_dim, element_bytes=4, batch=1):
        if any(type(x) is not int or x <= 0
               for x in (tokens, kv_heads, head_dim, element_bytes, batch)):
            raise ValueError("Cache dimensions must be positive integers")
        g = len(self.global_layers)
        slots = g * tokens + (self.layers - g) * min(tokens, self.window)
        return 2 * batch * kv_heads * head_dim * element_bytes * slots


# These names are stable experimental treatments, not a ranking.
ARMS = {
    "A0_global": (None, 0, False),
    "A1_local1024": (1024, 0, False),
    "A2_local512": (512, 0, False),
    "A3_sink512": (512, 4, False),
    "A4_global_gate": (None, 0, True),
    "A5_local512_gate": (512, 0, True),
    "A6_sink512_gate": (512, 4, True),
    "A7_sink1024": (1024, 4, False),
    "A8_all_local512": (512, 0, False),
    "A9_frontloaded512": (512, 0, False),
    "A10_all_local_sink512": (512, 4, False),
}


def arm_spec(name, layers, *, block_size=128, backend="math"):
    if name not in ARMS:
        raise ValueError(f"Unknown arm: {name}")
    window, sinks, gate = ARMS[name]
    # For non-multiples of four, explicitly retain the final aggregation layer.
    # This is not an exact Spark topology for those depths; record the full tuple.
    glob = tuple(range(layers)) if window is None else tuple(
        i for i in range(layers) if (i + 1) % 4 == 0 or i == layers - 1
    )
    if name in ("A8_all_local512", "A10_all_local_sink512"):
        # Mechanistic controls with no global layer at all, deliberately NOT
        # compute-matched to A2/A3. A10 is A8 plus the four prefix sinks: it
        # completes the globals x sinks square, whose sinks-absent row was
        # unmeasurable because every arm in it sat at exactly 0.000.
        glob = ()
    elif name == "A9_frontloaded512":
        # Same global count and final aggregation layer, no importance selection.
        glob = tuple(range(len(glob) - 1)) + (layers - 1,)
    return Spec(layers, glob, window or 512, sinks, gate, block_size, backend)
