from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import Constraint


@dataclass
class ConstraintSuite:
    """
    Runs multiple constraints and returns a dict of margins.

    Notes:
    - Margins are keyed by constraint.name (must be stable).
    - If a constraint raises unexpectedly, we catch it and assign a large negative margin.
      This keeps RL runs from crashing (academic requirement).
    """

    constraints: List[Constraint]
    fail_margin: float = -1e6  # used if a constraint throws an exception

    def metadata(self) -> Dict[str, Any]:
        """Metadata for run manifests/logging."""
        return {
            "type": "ConstraintSuite",
            "fail_margin": float(self.fail_margin),
            "constraints": [c.metadata() for c in self.constraints],
        }

    def evaluate_all(
        self, mol: Any, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        margins: Dict[str, float] = {}
        for c in self.constraints:
            key = c.name
            try:
                m = float(c.evaluate(mol, context=context))
            except Exception:
                # Failure-safe: do not crash training.
                # Keep a debug hint by suffixing the name (optional) or store elsewhere later.
                m = float(self.fail_margin)
            margins[key] = m
        return margins

    @staticmethod
    def satisfaction_rate(margins: Dict[str, float]) -> float:
        """Fraction of constraints satisfied (margin >= 0)."""
        if not margins:
            return 1.0
        ok = sum(1 for m in margins.values() if float(m) >= 0.0)
        return ok / float(len(margins))

    @staticmethod
    def violation_sum(margins: Dict[str, float]) -> float:
        """Sum of violation magnitudes: sum(max(0, -margin))."""
        s = 0.0
        for m in margins.values():
            mm = float(m)
            if mm < 0.0:
                s += -mm
        return s
