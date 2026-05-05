from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from rdkit.Chem import rdMolDescriptors

from .base import Constraint


def _bound_margin(
    x: float, min_val: Optional[float], max_val: Optional[float]
) -> Tuple[float, Dict[str, float]]:
    parts: Dict[str, float] = {}
    candidates = []

    if min_val is not None:
        mmin = float(x) - float(min_val)
        parts["min"] = mmin
        candidates.append(mmin)
    if max_val is not None:
        mmax = float(max_val) - float(x)
        parts["max"] = mmax
        candidates.append(mmax)

    if not candidates:
        return 1e9, parts  # no bounds => always satisfied

    return float(min(candidates)), parts


@dataclass(frozen=True)
class SizeConstraint(Constraint):
    min_heavy_atoms: Optional[int] = None
    max_heavy_atoms: Optional[int] = None

    min_rings: Optional[int] = None
    max_rings: Optional[int] = None

    min_rotors: Optional[int] = None
    max_rotors: Optional[int] = None

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "min_heavy_atoms": self.min_heavy_atoms,
            "max_heavy_atoms": self.max_heavy_atoms,
            "min_rings": self.min_rings,
            "max_rings": self.max_rings,
            "min_rotors": self.min_rotors,
            "max_rotors": self.max_rotors,
        }

    def evaluate(self, mol: Any, context: Optional[Dict[str, Any]] = None) -> float:
        if mol is None:
            return -1e6

        try:
            heavy = float(mol.GetNumHeavyAtoms())
            rings = float(rdMolDescriptors.CalcNumRings(mol))
            rotors = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
        except Exception:
            return -1e6

        m_heavy, _ = _bound_margin(heavy, self.min_heavy_atoms, self.max_heavy_atoms)
        m_rings, _ = _bound_margin(rings, self.min_rings, self.max_rings)
        m_rot, _ = _bound_margin(rotors, self.min_rotors, self.max_rotors)

        return float(min(m_heavy, m_rings, m_rot))
