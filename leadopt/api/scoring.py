from __future__ import annotations

"""Utilities to standardize scoring outputs in API return objects.

Phase 2.5 contract:
- objective: float (higher is better)
- components: dict[str, float] | None (floats only)
- metadata: dict[str, Any] (JSON-safe, may include constraints/validity/timings/raw)
"""

import json
from typing import Any, Dict, Optional, Tuple


def _floatify_dict(d: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        try:
            fv = float(v)
            # Drop NaN/Inf (non-portable in JSON)
            if fv != fv:
                continue
            if fv in (float("inf"), float("-inf")):
                continue
            out[str(k)] = fv
        except Exception:
            continue
    return out


def _json_sanitize(x: Any) -> Any:
    """Force JSON-serializable structure via best-effort string fallback."""
    try:
        return json.loads(json.dumps(x, default=str))
    except Exception:
        return str(x)


def score_to_fields(
    score: Any,
) -> Tuple[float, Optional[Dict[str, float]], Dict[str, Any]]:
    """Convert internal score objects into (objective, components, metadata).

    Supported inputs:
    - leadopt.scoring.types.ScoringResult
    - dict-like with objective/components/metadata/constraints/valid/fail_reason
    - objects with attributes .objective/.components/.metadata/.constraints/.valid/.fail_reason
    - fallback: try keys score/reward, else objective=0.0

    Returns:
      objective: float
      components: dict[str,float] | None
      metadata: dict[str,Any] (JSON-safe)
    """
    if score is None:
        return 0.0, None, {}

    # dict-like
    if isinstance(score, dict):
        objective = score.get("objective", None)
        if objective is None:
            objective = score.get("score", None)
        if objective is None:
            objective = score.get("reward", None)
        try:
            obj_f = float(objective) if objective is not None else 0.0
        except Exception:
            obj_f = 0.0

        comps = _floatify_dict(score.get("components", {}))
        constraints = _floatify_dict(score.get("constraints", {}))
        md = {
            "valid": bool(score.get("valid", True)),
            "fail_reason": score.get("fail_reason", None),
            "constraints": constraints,
            "metadata": _json_sanitize(score.get("metadata", {})),
        }
        return obj_f, (comps if comps else None), md

    # attribute-based (e.g., ScoringResult)
    objective = getattr(score, "objective", None)
    if objective is None:
        objective = getattr(score, "score", None)
    if objective is None:
        objective = getattr(score, "reward", None)

    try:
        obj_f = float(objective) if objective is not None else 0.0
    except Exception:
        obj_f = 0.0

    comps = _floatify_dict(getattr(score, "components", {}))
    constraints = _floatify_dict(getattr(score, "constraints", {}))
    md = {
        "valid": bool(getattr(score, "valid", True)),
        "fail_reason": getattr(score, "fail_reason", None),
        "constraints": constraints,
        "metadata": _json_sanitize(getattr(score, "metadata", {})),
    }
    return obj_f, (comps if comps else None), md


__all__ = ["score_to_fields"]
