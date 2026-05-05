from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from rdkit import Chem

from .rdkit_utils import (
    ComplexityMetrics,
    assert_valid_mol,
    canonical_smiles,
    mol_from_smiles,
)


@dataclass
class MoleculeState:
    """
    Lightweight container for the molecule and episode metadata.
    RL-specific fields are intentionally NOT included here.
    """

    mol: Chem.Mol
    step: int = 0
    max_steps: int = 10

    # arbitrary per-state metadata (trajectory logging, provenance, etc.)
    info: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.info is None:
            self.info = {}
        assert_valid_mol(self.mol)

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        *,
        step: int = 0,
        max_steps: int = 10,
        info: Optional[Dict[str, Any]] = None,
    ) -> "MoleculeState":
        mol = mol_from_smiles(smiles, sanitize=True)
        st = cls(mol=mol, step=step, max_steps=max_steps, info=info or {})
        return st

    @property
    def smiles(self) -> str:
        return canonical_smiles(self.mol)

    @property
    def complexity(self) -> ComplexityMetrics:
        return ComplexityMetrics.compute(self.mol)
