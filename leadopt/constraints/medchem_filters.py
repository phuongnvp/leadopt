from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdmolops

from .base import Constraint


def _bound_margin(
    x: float, min_val: Optional[float], max_val: Optional[float]
) -> Tuple[float, Dict[str, float]]:
    """
    Returns (combined_margin, parts) with the same convention as SizeConstraint:
      - if min_val: margin_min = x - min_val
      - if max_val: margin_max = max_val - x
      - combined margin is min(applicable margins)
    """
    parts: Dict[str, float] = {}
    candidates: List[float] = []

    if min_val is not None:
        mmin = float(x) - float(min_val)
        parts["min"] = mmin
        candidates.append(mmin)
    if max_val is not None:
        mmax = float(max_val) - float(x)
        parts["max"] = mmax
        candidates.append(mmax)

    if not candidates:
        return 1e9, parts

    return float(min(candidates)), parts


@dataclass(frozen=True)
class ChargeConstraint(Constraint):
    """
    Constraint on total formal charge.
    Example: allowed_range = [-1, +1] for medchem sanity.
    """

    min_charge: Optional[int] = -1
    max_charge: Optional[int] = 1

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "min_charge": self.min_charge,
            "max_charge": self.max_charge,
        }

    def evaluate(self, mol: Any, context: Optional[Dict[str, Any]] = None) -> float:
        if mol is None:
            return -1e6
        try:
            # RDKit formal charge for molecule
            q = float(rdmolops.GetFormalCharge(mol))
        except Exception:
            return -1e6

        m, _ = _bound_margin(q, self.min_charge, self.max_charge)
        return float(m)


@dataclass(frozen=True)
class ElementConstraint(Constraint):
    """
    Require all atom symbols to be in allowlist (common medchem elements by default).
    Severity = number of disallowed atoms (margin = -count).
    """

    allowed_elements: Tuple[str, ...] = (
        "H",
        "C",
        "N",
        "O",
        "S",
        "F",
        "Cl",
        "Br",
        "I",
        "P",
    )

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "allowed_elements": list(self.allowed_elements),
        }

    def evaluate(self, mol: Any, context: Optional[Dict[str, Any]] = None) -> float:
        if mol is None:
            return -1e6
        allowed = set(self.allowed_elements)
        try:
            bad = 0
            for a in mol.GetAtoms():
                sym = a.GetSymbol()
                if sym not in allowed:
                    bad += 1
        except Exception:
            return -1e6

        if bad == 0:
            return 1.0
        return -float(bad)


@dataclass(frozen=True)
class RingCountConstraint(Constraint):
    """
    Constraint on total ring count (rdMolDescriptors.CalcNumRings).
    """

    max_rings: Optional[int] = 8
    min_rings: Optional[int] = None

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "min_rings": self.min_rings,
            "max_rings": self.max_rings,
        }

    def evaluate(self, mol: Any, context: Optional[Dict[str, Any]] = None) -> float:
        if mol is None:
            return -1e6
        try:
            rings = float(rdMolDescriptors.CalcNumRings(mol))
        except Exception:
            return -1e6

        m, _ = _bound_margin(rings, self.min_rings, self.max_rings)
        return float(m)


@dataclass(frozen=True)
class HBDHBAConstraint(Constraint):
    """
    Constraint on H-bond donors/acceptors.
    Uses rdMolDescriptors.CalcNumHBD / CalcNumHBA.
    """

    max_hbd: Optional[int] = 3
    max_hba: Optional[int] = 8
    min_hbd: Optional[int] = None
    min_hba: Optional[int] = None

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "min_hbd": self.min_hbd,
            "max_hbd": self.max_hbd,
            "min_hba": self.min_hba,
            "max_hba": self.max_hba,
        }

    def evaluate(self, mol: Any, context: Optional[Dict[str, Any]] = None) -> float:
        if mol is None:
            return -1e6
        try:
            hbd = float(rdMolDescriptors.CalcNumHBD(mol))
            hba = float(rdMolDescriptors.CalcNumHBA(mol))
        except Exception:
            return -1e6

        m_hbd, _ = _bound_margin(hbd, self.min_hbd, self.max_hbd)
        m_hba, _ = _bound_margin(hba, self.min_hba, self.max_hba)
        return float(min(m_hbd, m_hba))


@dataclass(frozen=True)
class ReactiveGroupConstraint(Constraint):
    """
    Ban explicit reactive/unstable functional groups using a small SMARTS blacklist.

    Default list is intentionally minimal and explicit (Tier 1). Add new patterns only by
    introducing a new version (reactive_smarts_v2, etc.) or by passing patterns explicitly.
    """

    patterns: Tuple[Tuple[str, str], ...] = (
        # name, smarts
        ("acyl_halide", "[CX3](=O)[Cl,Br,I]"),
        ("sulfonyl_halide", "[SX4](=O)(=O)[Cl,Br,I]"),
        ("isocyanate", "N=C=O"),
        ("peroxide", "[OX2][OX2]"),
        ("azide", "[N-]=[N+]=N"),
        ("diazo", "[N]=[N+]=[C-]"),
    )

    library_version: str = "reactive_smarts_v1"
    _compiled: Tuple[Tuple[str, Chem.Mol], ...] = field(
        default_factory=tuple, init=False, repr=False
    )

    def __post_init__(self) -> None:
        compiled: List[Tuple[str, Chem.Mol]] = []
        for name, smarts in self.patterns:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                raise ValueError(
                    f"Invalid SMARTS in ReactiveGroupConstraint: {name}={smarts!r}"
                )
            compiled.append((str(name), patt))
        object.__setattr__(self, "_compiled", tuple(compiled))

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "library_version": self.library_version,
            "patterns": [{"name": n, "smarts": s} for (n, s) in self.patterns],
        }

    def evaluate(self, mol: Any, context: Optional[Dict[str, Any]] = None) -> float:
        if mol is None:
            return -1e6

        try:
            hits = 0
            for _, patt in self._compiled:
                if mol.HasSubstructMatch(patt):
                    hits += 1
        except Exception:
            return -1e6

        if hits == 0:
            return 1.0
        return -float(hits)
