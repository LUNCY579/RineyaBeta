"""Significance-Gated Quantized Remnant (SGQR) reference implementation.

SGQR isolates the frozen low-bit donor and significance-routing hypotheses. It
contains no next-token confidence gate; confidence can be evaluated later as an
optional budget controller on top of this core.
"""

from .install import install_sgqr_mlp, restore_original_mlp
from .quantization import PackedSymmetricMatrix
from .remnant import (
    DynamicUtilityRouter,
    FrozenQuantizedMLPRemnant,
    SGQRMLPWrapper,
)
from .utility import marginal_recovery_utility, recovery_ratio

__all__ = [
    "DynamicUtilityRouter",
    "FrozenQuantizedMLPRemnant",
    "PackedSymmetricMatrix",
    "SGQRMLPWrapper",
    "install_sgqr_mlp",
    "marginal_recovery_utility",
    "recovery_ratio",
    "restore_original_mlp",
]
