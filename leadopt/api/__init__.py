"""Stable importable Python API for leadopt.

Phase 2.1 (API skeleton):
- Provide import paths and stable function names.
- Heavy dependencies must not be imported at module import time.

The CLI remains the source-of-truth. In Phase 2.3+, these functions will
call shared engine implementations extracted from the CLI.
"""

from __future__ import annotations

from .beam import beam  # noqa: F401
from .generate import generate  # noqa: F401
from .run import run  # noqa: F401
from .train import train  # noqa: F401
from .types import (  # noqa: F401
    ActionStep,
    ActionTrace,
    BeamResult,
    GenerateResult,
    MoleculeRecord,
    RunMetadata,
    RunResult,
    TrainResult,
)

__all__ = [
    # Types
    "MoleculeRecord",
    "ActionStep",
    "ActionTrace",
    "RunMetadata",
    "RunResult",
    "BeamResult",
    "GenerateResult",
    "TrainResult",
    # Functions
    "run",
    "beam",
    "generate",
    "train",
]
