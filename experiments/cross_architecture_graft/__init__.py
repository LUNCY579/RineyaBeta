"""Cross-architecture graft experiment package.

The package is deliberately isolated from ``scripts/train`` so adding the
experiment cannot invalidate historical checkpoint fingerprints.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
