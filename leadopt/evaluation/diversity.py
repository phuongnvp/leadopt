from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


@dataclass(frozen=True)
class DiversityMetrics:
    n: int
    mean_pairwise_tanimoto: float
    median_pairwise_tanimoto: float
    mean_novelty_vs_lead: float  # 1 - Tanimoto(lead, mol)
    unique_smiles: int


def _fp(m: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)


def compute_diversity_metrics(
    mols: Sequence[Chem.Mol],
    lead: Optional[Chem.Mol] = None,
    radius: int = 2,
    n_bits: int = 2048,
    max_pairs: int = 20000,
) -> DiversityMetrics:
    """
    Deterministic. Computes pairwise Tanimoto among mols (subsample pairs if large),
    and novelty vs lead if provided.
    """
    clean = [m for m in mols if m is not None]
    n = len(clean)
    if n == 0:
        return DiversityMetrics(0, float("nan"), float("nan"), float("nan"), 0)

    fps = [_fp(m, radius=radius, n_bits=n_bits) for m in clean]

    # Unique SMILES (canonical)
    smiles = [Chem.MolToSmiles(m, canonical=True) for m in clean]
    unique_smiles = len(set(smiles))

    # Pairwise similarities (upper triangle)
    sims: List[float] = []
    # If too many molecules, cap number of pairs deterministically
    # (first K pairs in lexicographic index order)
    total_pairs = n * (n - 1) // 2
    cap = min(total_pairs, max_pairs)

    count = 0
    for i in range(n):
        if count >= cap:
            break
        for j in range(i + 1, n):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            count += 1
            if count >= cap:
                break

    mean_pair = float(sum(sims) / len(sims)) if sims else float("nan")
    median_pair = float(sorted(sims)[len(sims) // 2]) if sims else float("nan")

    novelty = float("nan")
    if lead is not None:
        lead_fp = _fp(lead, radius=radius, n_bits=n_bits)
        nov = [1.0 - float(DataStructs.TanimotoSimilarity(lead_fp, fp)) for fp in fps]
        novelty = float(sum(nov) / len(nov))

    return DiversityMetrics(
        n=n,
        mean_pairwise_tanimoto=mean_pair,
        median_pairwise_tanimoto=median_pair,
        mean_novelty_vs_lead=novelty,
        unique_smiles=unique_smiles,
    )
