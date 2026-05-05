from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from rdkit import Chem
from rdkit.Chem import Lipinski


@dataclass(frozen=True)
class ComplexityBreakdown:
    heavy_atoms: int
    rings: int
    stereocenters: int
    rotors: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "heavy_atoms": int(self.heavy_atoms),
            "rings": int(self.rings),
            "stereocenters": int(self.stereocenters),
            "rotors": int(self.rotors),
        }


def compute_complexity_breakdown(mol: Chem.Mol) -> ComplexityBreakdown:
    """
    Deterministic complexity proxies based on RDKit.

    These are intentionally simple and stable across RDKit versions:
      - heavy atom count
      - ring count
      - stereocenter count (unassigned included)
      - rotatable bond count

    Raises:
      - ValueError if mol is None
    """
    if mol is None:
        raise ValueError("mol is None")

    heavy_atoms = int(mol.GetNumHeavyAtoms())
    rings = int(mol.GetRingInfo().NumRings())

    # includeUnassigned=True makes this stable even if chirality not assigned
    stereocenters = int(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)))

    # Lipinski rotors is widely used and deterministic
    rotors = int(Lipinski.NumRotatableBonds(mol))

    return ComplexityBreakdown(
        heavy_atoms=heavy_atoms,
        rings=rings,
        stereocenters=stereocenters,
        rotors=rotors,
    )


def compute_complexity_score(mol: Chem.Mol) -> float:
    """
    A single scalar complexity score used for reward shaping.

    Weighting is conservative and interpretable; you can tune later without
    changing the interface.

    Higher score => more complex.

    Current weights:
      1.0 * heavy atoms
      5.0 * rings
      3.0 * stereocenters
      0.5 * rotors
    """
    b = compute_complexity_breakdown(mol)
    return 1.0 * b.heavy_atoms + 5.0 * b.rings + 3.0 * b.stereocenters + 0.5 * b.rotors
