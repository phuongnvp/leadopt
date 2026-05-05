from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import Scorer
from .types import ScoringResult


def _extract_compute_cost_units(md: Dict[str, Any]) -> float:
    """Prefer compute_cost_units, then compute_cost, then compute_cost_s."""
    if not isinstance(md, dict):
        return 0.0
    for k in ("compute_cost_units", "compute_cost", "compute_cost_s"):
        if k in md:
            try:
                return float(md[k])
            except Exception:
                continue
    return 0.0


@dataclass
class CompositeScorer(Scorer):
    """Multi-objective scorer that aggregates multiple sub-scorers into one scalar objective.

    Preserves frozen contracts by returning a single scalar `ScoringResult.objective`,
    while logging per-objective values in components and per-scorer results in metadata.

    MVP aggregation mode: weighted_sum.
    """

    # Ordered list of (name, scorer) to preserve deterministic evaluation order.
    scorers: Sequence[Tuple[str, Scorer]]

    aggregation_mode: str = "weighted_sum"
    weights: Optional[Dict[str, float]] = None
    normalize: bool = False  # reserved for future extensions

    # Allow YAML override while keeping the Scorer base default.
    fail_objective: float = Scorer.fail_objective

    # Stable version string for reproducibility artifacts.
    version: str = "0"

    extra_metadata: Optional[Dict[str, Any]] = None

    def scorer_metadata(self) -> Dict[str, Any]:
        md: Dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "type": "composite",
            "aggregation": {
                "mode": str(self.aggregation_mode),
                "weights": dict(self.weights or {}),
                "normalize": bool(self.normalize),
            },
            "sub_scorers": [
                {
                    "name": str(name),
                    "type": scorer.name,
                    "version": getattr(scorer, "version", "0"),
                    "metadata": dict(scorer.scorer_metadata() or {}),
                }
                for name, scorer in self.scorers
            ],
        }
        if self.extra_metadata:
            md.update(dict(self.extra_metadata))
        return md

    def _validate_config(self) -> Optional[str]:
        if not isinstance(self.scorers, Sequence) or len(self.scorers) == 0:
            return "input:no_sub_scorers"
        names = [n for n, _ in self.scorers]
        if len(names) != len(set(names)):
            return "input:duplicate_sub_scorer_names"

        if str(self.aggregation_mode).strip().lower() != "weighted_sum":
            return f"input:unknown_aggregation_mode:{self.aggregation_mode}"

        w = self.weights or {}
        for n in names:
            if n not in w:
                return f"input:missing_weight:{n}"
        for k in w.keys():
            if k not in set(names):
                return f"input:unknown_weight_key:{k}"
        return None

    def score(
        self, mol: Any, context: Optional[Dict[str, Any]] = None
    ) -> ScoringResult:
        try:
            reason = self._validate_config()
            if reason is not None:
                md = {**self.scorer_metadata()}
                if context:
                    md["context"] = dict(context)
                return ScoringResult(
                    objective=float(self.fail_objective),
                    components={"objective": float(self.fail_objective)},
                    valid=False,
                    fail_reason=reason,
                    metadata=md,
                )

            mode = str(self.aggregation_mode).strip().lower()
            weights = dict(self.weights or {})

            sub_results: List[Dict[str, Any]] = []
            components: Dict[str, float] = {}

            total_cost_units = 0.0
            aggregate = 0.0
            all_valid = True

            base_ctx = dict(context) if context else {}
            for name, scorer in self.scorers:
                sub_ctx = dict(base_ctx)
                sub_ctx["_composite_objective_name"] = str(name)

                r = scorer.score(mol, context=sub_ctx)

                w = float(weights.get(name, 0.0))
                obj = float(r.objective)

                # Keep failure-safe: if a sub-scorer is invalid, keep composite invalid.
                # Use a low fallback if the sub-scorer returned a non-low value.
                if not r.valid:
                    all_valid = False
                    obj = float(obj if obj <= 0.0 else self.fail_objective)

                contrib = w * obj

                components[f"obj:{name}"] = float(obj)
                components[f"weight:{name}"] = float(w)
                components[f"contrib:{name}"] = float(contrib)

                aggregate += contrib
                total_cost_units += _extract_compute_cost_units(r.metadata)

                sub_results.append(
                    {
                        "name": str(name),
                        "type": scorer.name,
                        "version": getattr(scorer, "version", "0"),
                        "objective": float(r.objective),
                        "valid": bool(r.valid),
                        "fail_reason": r.fail_reason,
                        "metadata": dict(r.metadata),
                    }
                )

            if mode != "weighted_sum":
                all_valid = False

            components["objective"] = float(aggregate)
            components["compute_cost_units"] = float(total_cost_units)
            components["compute_cost"] = float(total_cost_units)

            md = {
                **self.scorer_metadata(),
                "sub_results": sub_results,
                "compute_cost_units": float(total_cost_units),
                "compute_cost": float(total_cost_units),
            }
            if context:
                md["context"] = dict(context)

            fail_reason = None if all_valid else "input:one_or_more_subscores_invalid"
            return ScoringResult(
                objective=float(aggregate),
                components=components,
                valid=bool(all_valid),
                fail_reason=fail_reason,
                metadata=md,
            )

        except Exception as e:
            md = {
                **self.scorer_metadata(),
                "exception_type": type(e).__name__,
                "exception_message": str(e),
            }
            if context:
                md["context"] = dict(context)
            return ScoringResult(
                objective=float(self.fail_objective),
                components={"objective": float(self.fail_objective)},
                valid=False,
                fail_reason=f"exception:{type(e).__name__}",
                metadata=md,
            )
