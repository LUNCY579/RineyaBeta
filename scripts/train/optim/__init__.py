"""Optimizer and effective-delta geometry primitives for RGE-PEFT."""

from .factory import build_optimizer, optimizer_contract
from .muon import Muon, get_muon_scale_factor, newton_schulz_quintic

__all__ = [
    "Muon",
    "build_optimizer",
    "get_muon_scale_factor",
    "newton_schulz_quintic",
    "optimizer_contract",
]
