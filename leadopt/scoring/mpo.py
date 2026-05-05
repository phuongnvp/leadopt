from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, rdMolDescriptors

from .base import Scorer
from .types import ScoringResult


def _compute_property(mol: Any, name: str) -> float:
    """
    Deterministic RDKit property computation.

    Keep this minimal and stable. We do NOT normalize by default (raw properties),
    because different projects want different target regions / transforms.
    """
    key = name.strip().lower()

    if key == "qed":
        return float(QED.qed(mol))
    if key in {"mw", "molwt", "mol_wt"}:
        return float(Descriptors.MolWt(mol))
    if key in {"logp", "clogp"}:
        return float(Crippen.MolLogP(mol))
    if key == "tpsa":
        return float(rdMolDescriptors.CalcTPSA(mol))
    if key in {"hbd", "h_donors"}:
        return float(rdMolDescriptors.CalcNumHBD(mol))
    if key in {"hba", "h_acceptors"}:
        return float(rdMolDescriptors.CalcNumHBA(mol))
    if key in {"rotb", "rotors", "num_rot_bonds"}:
        return float(rdMolDescriptors.CalcNumRotatableBonds(mol))
    if key in {"rings", "ringcount"}:
        return float(rdMolDescriptors.CalcNumRings(mol))

    raise ValueError(f"unknown_property:{name}")


def _apply_transform(
    value: float, transform: Optional[Dict[str, Any]]
) -> Tuple[float, Dict[str, Any]]:
    """
    Apply an optional deterministic transform.

    Supported:
      - identity (default)
      - linear: y = slope * x + intercept
      - clamp: clamp to [min, max]
      - triangle: desirability around target with tolerance (linear falloff to 0)

    Returns (transformed_value, transform_metadata)
    """
    if not transform:
        return float(value), {"type": "identity"}

    ttype = str(transform.get("type", "identity")).strip().lower()
    params = dict(transform.get("params", {}) or {})

    x = float(value)

    if ttype == "identity":
        return x, {"type": "identity"}

    if ttype == "linear":
        slope = float(params.get("slope", 1.0))
        intercept = float(params.get("intercept", 0.0))
        return (slope * x + intercept), {
            "type": "linear",
            "slope": slope,
            "intercept": intercept,
        }

    if ttype == "clamp":
        lo = float(params.get("min", -1e18))
        hi = float(params.get("max", 1e18))
        y = max(lo, min(hi, x))
        return y, {"type": "clamp", "min": lo, "max": hi}

    if ttype == "triangle":
        # desirability in [0, 1], peaked at target, linear falloff over tolerance
        target = float(params["target"])
        tol = float(params.get("tolerance", 1.0))
        tol = max(tol, 1e-12)
        d = abs(x - target)
        y = max(0.0, 1.0 - (d / tol))
        return y, {"type": "triangle", "target": target, "tolerance": tol}

    raise ValueError(f"unknown_transform:{ttype}")


@dataclass
class MPOPropertySpec:
    name: str
    weight: float
    transform: Optional[Dict[str, Any]] = None


@dataclass
class MPOScorer(Scorer):
    """
    Multi-Property Optimization scorer.

    - Deterministic (RDKit descriptors only)
    - Failure-safe (returns valid=False with fail_reason)
    - Objective direction standardized: higher is better
    - Components include raw values, transformed values, and weighted contributions

    YAML schema (Phase 3.1):
      scoring:
        type: MPOScorer
        params:
          properties:
            - name: qed
              weight: 1.0
              transform: {type: identity, params: {}}
          aggregation: weighted_sum
    """

    properties: List[Dict[str, Any]]
    aggregation: str = "weighted_sum"
    fail_objective: float = Scorer.fail_objective
    version_override: str = "0"

    @property
    def version(self) -> str:
        return str(self.version_override)

    def _parse_specs(self) -> List[MPOPropertySpec]:
        specs: List[MPOPropertySpec] = []
        for p in self.properties:
            name = str(p["name"])
            weight = float(p["weight"])
            transform = p.get("transform", None)
            specs.append(MPOPropertySpec(name=name, weight=weight, transform=transform))
        return specs

    def scorer_metadata(self) -> Dict[str, Any]:
        # Keep metadata JSON-friendly and stable.
        return {
            "name": self.name,
            "version": self.version,
            "type": "mpo",
            "aggregation": str(self.aggregation),
            "properties": [
                {
                    "name": str(p.get("name")),
                    "weight": float(p.get("weight")),
                    "transform": (
                        dict(p.get("transform"))
                        if p.get("transform") is not None
                        else None
                    ),
                }
                for p in (self.properties or [])
            ],
        }

    def score(
        self, mol: Any, context: Optional[Dict[str, Any]] = None
    ) -> ScoringResult:
        try:
            if mol is None:
                return ScoringResult(
                    objective=float(self.fail_objective),
                    components={"objective": float(self.fail_objective)},
                    valid=False,
                    fail_reason="input:mol_is_none",
                    metadata={**self.scorer_metadata()},
                )

            if str(self.aggregation).strip().lower() != "weighted_sum":
                return ScoringResult(
                    objective=float(self.fail_objective),
                    components={"objective": float(self.fail_objective)},
                    valid=False,
                    fail_reason=f"input:unsupported_aggregation:{self.aggregation}",
                    metadata={**self.scorer_metadata()},
                )

            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

            specs = self._parse_specs()
            if len(specs) == 0:
                return ScoringResult(
                    objective=float(self.fail_objective),
                    components={"objective": float(self.fail_objective)},
                    valid=False,
                    fail_reason="input:no_properties",
                    metadata={**self.scorer_metadata(), "smiles": smiles},
                )

            components: Dict[str, float] = {}
            contrib_sum = 0.0

            # Compute properties deterministically, then aggregate.
            for spec in specs:
                raw = _compute_property(mol, spec.name)
                transformed, tmeta = _apply_transform(raw, spec.transform)

                w = float(spec.weight)
                contrib = w * float(transformed)

                # floats-only component logging
                key = spec.name.strip().lower()
                components[f"{key}_raw"] = float(raw)
                components[f"{key}"] = float(transformed)
                components[f"{key}_weight"] = float(w)
                components[f"{key}_contrib"] = float(contrib)

                contrib_sum += contrib

            objective = float(contrib_sum)
            components["objective"] = float(objective)

            md = {**self.scorer_metadata(), "smiles": smiles}
            if context:
                md["context"] = dict(context)

            return ScoringResult(
                objective=float(objective),
                components=components,
                valid=True,
                fail_reason=None,
                metadata=md,
            )

        except Exception as e:
            md = {**self.scorer_metadata()}
            md["exception_type"] = type(e).__name__
            md["exception_message"] = str(e)
            try:
                md["smiles"] = (
                    Chem.MolToSmiles(mol, isomericSmiles=True)
                    if mol is not None
                    else None
                )
            except Exception:
                md["smiles"] = None
            if context:
                md["context"] = dict(context)

            return ScoringResult(
                objective=float(self.fail_objective),
                components={"objective": float(self.fail_objective)},
                valid=False,
                fail_reason=f"exception:{type(e).__name__}",
                metadata=md,
            )
