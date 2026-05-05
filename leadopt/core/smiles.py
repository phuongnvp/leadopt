from __future__ import annotations

"""SMILES string helpers.

This module is import-light: it does not import RDKit at module import time.
"""

from typing import Optional

from . import _require_rdkit


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string.

    Returns
    -------
    Optional[str]
        Canonical SMILES if parsing succeeds, otherwise ``None``.

    Notes
    -----
    - Requires RDKit at runtime.
    - Uses RDKit canonicalization with ``canonical=True``.
    """

    _require_rdkit()
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def canonicalize_smiles_or_empty(smiles: str) -> str:
    """Canonicalize SMILES string, returning "" on failure.

    This is useful for legacy code paths that historically returned empty
    string instead of None.
    """

    out = canonicalize_smiles(smiles)
    return out or ""