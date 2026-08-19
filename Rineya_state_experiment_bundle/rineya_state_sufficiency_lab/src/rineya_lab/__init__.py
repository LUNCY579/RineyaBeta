"""Rineya State Sufficiency Lab.

A controlled research harness comparing two supervision representations trained
from one immutable base checkpoint:

FULL   : problem -> complete reasoning trajectory and answer
SPLIT  : (problem, compact current state) -> (next step, updated state, done)

The SPLIT records are globally shuffled: the optimizer never relies on seeing
transition ``t-1`` immediately before transition ``t``.

Nothing at import time requires torch. Training modules import their heavy
dependencies lazily so that data preparation, evaluation with the reference
backend, analysis and the full test suite run on a CPU-only machine.
"""

from rineya_lab.version import (
    DATASET_CONTRACT_VERSION,
    SCHEMA_VERSION,
    __version__,
)

__all__ = ["__version__", "SCHEMA_VERSION", "DATASET_CONTRACT_VERSION"]
