from __future__ import annotations

from typing import Dict

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


def compute_props(mol: Chem.Mol) -> Dict[str, float]:
    """
    A compact medchem descriptor panel. Extend later as needed.
    """
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HeavyAtoms": float(mol.GetNumHeavyAtoms()),
        "Rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "RotB": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
    }


def delta_props(lead: Dict[str, float], final: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(set(lead.keys()) | set(final.keys()))
    return {f"d{k}": float(final.get(k, 0.0) - lead.get(k, 0.0)) for k in keys}
