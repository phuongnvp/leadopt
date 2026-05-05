from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ScoringResult:
    """
    Structured scoring output.

    Conventions:
    - objective: float, higher is always better.
    - components: numeric breakdown of objective (floats only for reproducibility).
    - constraints: constraint margins (positive OK, negative violation).
    - valid: if False, objective/components may be a fallback (e.g., fail_objective).
    - fail_reason: short reason string if valid is False.
    - metadata: any extra info (timings, cache hits, raw docking energy, etc.).
                This may contain non-floats; keep components numeric.
    """

    objective: float
    components: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, float] = field(default_factory=dict)
    valid: bool = True
    fail_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_constraints(self, constraints: Dict[str, float]) -> "ScoringResult":
        """Return a copy with constraints merged/overwritten by provided dict."""
        merged = dict(self.constraints)
        merged.update(constraints)
        return ScoringResult(
            objective=self.objective,
            components=dict(self.components),
            constraints=merged,
            valid=self.valid,
            fail_reason=self.fail_reason,
            metadata=dict(self.metadata),
        )

    def summary(self) -> Dict[str, Any]:
        """Small JSON-serializable summary used for logs/trajectory records."""

        def _floatify(d: Dict[str, Any]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            for k, v in (d or {}).items():
                try:
                    fv = float(v)
                    # Drop NaN/Inf to keep artifacts numeric and stable.
                    # (JSON allows NaN in some implementations, but it's not portable.)
                    if fv != fv:  # NaN check
                        continue
                    if fv in (float("inf"), float("-inf")):
                        continue
                    out[str(k)] = fv
                except Exception:
                    # Keep failure-safe: if a component/constraint is malformed,
                    # omit it rather than crashing logging.
                    continue
            return out

        return {
            "objective": float(self.objective),
            # Enforce the "floats only" convention in serialized artifacts.
            # (Internally, scorers should still strive to store floats.)
            "components": _floatify(self.components),
            "constraints": _floatify(self.constraints),
            "valid": bool(self.valid),
            "fail_reason": self.fail_reason,
            "metadata": dict(self.metadata),
        }
