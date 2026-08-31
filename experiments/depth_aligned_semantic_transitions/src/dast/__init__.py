"""Depth-Aligned Semantic Transitions (DAST).

DAST supervises semantic reasoning milestones at ordered Transformer depths while
keeping the final language-model objective intact.  The package is deliberately
an experiment harness: a green smoke test is engineering evidence, not a model-
quality result.
"""

from dast.config import DASTConfig as DASTConfig
from dast.config import load_config as load_config

__all__ = ["DASTConfig", "load_config"]
__version__ = "0.1.0"
