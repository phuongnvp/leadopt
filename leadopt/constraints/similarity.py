from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from .base import Constraint

RDLogger.DisableLog("rdApp.*")
FingerprintType = Literal["morgan"]


def _morgan_fp(mol: Chem.Mol, radius: int, nbits: int):
    # Use explicit config for reproducibility
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


@dataclass(frozen=True)
class SimilarityConstraint(Constraint):
    """
    Similarity constraint to a lead/reference molecule.

    Uses Tanimoto similarity on explicit fingerprints (default Morgan r=2, nBits=2048).

    Supports:
      - min_sim: require similarity >= min_sim
      - max_sim: require similarity <= max_sim  (useful to enforce novelty)
    Margin rules:
      - min bound: sim - min_sim
      - max bound: max_sim - sim
    Combined margin = min(applicable margins)
    """

    lead_smiles: str
    min_sim: Optional[float] = None
    max_sim: Optional[float] = None

    fp_type: FingerprintType = "morgan"
    radius: int = 2
    nbits: int = 2048

    def __post_init__(self):
        # Precompute lead fingerprint for speed; fail early if lead is invalid
        lead = Chem.MolFromSmiles(self.lead_smiles)
        if lead is None:
            raise ValueError(
                f"Invalid lead_smiles for SimilarityConstraint: {self.lead_smiles!r}"
            )
        object.__setattr__(self, "_lead_fp", _morgan_fp(lead, self.radius, self.nbits))

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "lead_smiles": self.lead_smiles,
            "min_sim": self.min_sim,
            "max_sim": self.max_sim,
            "fp_type": self.fp_type,
            "radius": self.radius,
            "nbits": self.nbits,
        }

    def evaluate(self, mol: Any, context: Optional[Dict[str, Any]] = None) -> float:
        if mol is None:
            return -1e6

        try:
            fp = _morgan_fp(mol, self.radius, self.nbits)
            sim = float(DataStructs.TanimotoSimilarity(fp, self._lead_fp))
        except Exception:
            return -1e6

        margins = []

        if self.min_sim is not None:
            margins.append(sim - float(self.min_sim))
        if self.max_sim is not None:
            margins.append(float(self.max_sim) - sim)

        if not margins:
            return 1e9

        return float(min(margins))
