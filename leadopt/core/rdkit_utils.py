from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import rdchem, rdMolDescriptors

from .errors import ChemistryError


def mol_from_smiles(smiles: str, sanitize: bool = True) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        raise ChemistryError(f"Invalid SMILES: {smiles!r}")
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            raise ChemistryError(
                f"Sanitization failed for SMILES {smiles!r}: {e}"
            ) from e
    return mol


def canonical_smiles(mol: Chem.Mol) -> str:
    # isomericSmiles True gives stable stereochemistry output
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def clone_mol(mol: Chem.Mol) -> Chem.Mol:
    return Chem.Mol(mol)


def sanitize_or_raise(mol: Chem.Mol) -> Chem.Mol:
    try:
        Chem.SanitizeMol(mol)
        return mol
    except rdchem.KekulizeException:
        # Fallback: accept aromatic form even if Kekulé form can't be assigned
        try:
            flags = (
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            )
            Chem.SanitizeMol(mol, sanitizeOps=flags)
            return mol
        except Exception as e2:
            raise ChemistryError(
                f"Sanitization failed (even without kekulize): {e2}"
            ) from e2
    except Exception as e:
        raise ChemistryError(f"Sanitization failed: {e}") from e


def compile_substructure_query(pattern: str, *, kind: str = "auto") -> Chem.Mol:
    """
    kind: 'smarts', 'smiles', or 'auto'
    """
    q = None
    if kind in ("smarts", "auto"):
        q = Chem.MolFromSmarts(pattern)
    if q is None and kind in ("smiles", "auto"):
        q = Chem.MolFromSmiles(pattern)
        if q is not None:
            # Optional: make SMILES-core behave more like a substructure query (less brittle)
            # This keeps atom/bond types but avoids over-constraining some properties.
            q = Chem.AdjustQueryProperties(q)
    if q is None:
        raise ValueError(f"Could not parse core pattern as {kind}: {pattern}")
    return q


def is_connected(mol: Chem.Mol) -> bool:
    # One fragment => connected
    frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    return len(frags) == 1


@dataclass(frozen=True)
class ComplexityMetrics:
    heavy_atoms: int
    rings: int
    rot_bonds: int
    hetero_atoms: int

    @staticmethod
    def compute(mol: Chem.Mol) -> "ComplexityMetrics":
        heavy = mol.GetNumHeavyAtoms()
        rings = rdMolDescriptors.CalcNumRings(mol)
        rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
        return ComplexityMetrics(
            heavy_atoms=heavy, rings=rings, rot_bonds=rot, hetero_atoms=hetero
        )


def assert_valid_mol(mol: Chem.Mol, require_connected: bool = True) -> None:
    if mol is None:
        raise ChemistryError("Mol is None")
    sanitize_or_raise(mol)
    if require_connected and not is_connected(mol):
        raise ChemistryError("Molecule is disconnected (multiple fragments).")


def atom_indices(mol: Chem.Mol) -> list[int]:
    return list(range(mol.GetNumAtoms()))


def bond_indices(mol: Chem.Mol) -> list[int]:
    return list(range(mol.GetNumBonds()))
