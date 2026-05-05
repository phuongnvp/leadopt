"""Internal execution engines used by both CLI and the public Python API.

Design contract (Phase 2.3):
- Engines hold shared implementation.
- CLI wrappers should be thin and call engines.
- Engines should not depend on CLI modules.
"""

from __future__ import annotations

__all__ = [
    "beam_search",
    "run_rollout",
]
