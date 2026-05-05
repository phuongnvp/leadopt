"""
leadopt.sar.context

Deterministic local-context keys for stratified (quasi-matched) SAR.

Rationale
---------
Global operator/template summaries are confounded by when/where edits are applied.
A lightweight way to reduce confounding (without changing the env) is to stratify
edit effects by a deterministic representation of the *local chemical context*
around the edited site on the *pre-edit* molecule.

This module provides:
- extraction of neighborhood atoms around a site
- a deterministic context key based on a Morgan fingerprint focused on those atoms

Notes
-----
- This is not a claim of causality; it's a stronger association by conditioning.
- Site conventions are inherited from logged actions (tuple of atom indices).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem


def _ensure_mol(smiles: str) -> Chem.Mol | None:
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def atoms_within_radius(
    mol: Chem.Mol, site_atoms: Sequence[int], radius: int
) -> List[int]:
    """Return sorted unique atom indices within `radius` bonds of any site atom."""
    if mol is None:
        return []
    n = mol.GetNumAtoms()
    seeds = [int(a) for a in site_atoms if 0 <= int(a) < n]
    if not seeds:
        return []

    dist = Chem.GetDistanceMatrix(mol)
    keep = set()
    for s in seeds:
        for i in range(n):
            if dist[s, i] <= radius:
                keep.add(int(i))
    return sorted(keep)


def focused_morgan_bits(
    mol: Chem.Mol,
    *,
    from_atoms: Sequence[int],
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[int, ...]:
    """Morgan bits focused on a local region via RDKit's `fromAtoms`.

    Returns a sorted tuple of on-bit indices (deterministic, JSON-safe).
    """
    if mol is None or not from_atoms:
        return tuple()

    bv = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        int(radius),
        nBits=int(n_bits),
        fromAtoms=[int(a) for a in from_atoms],
    )
    return tuple(sorted(int(i) for i in bv.GetOnBits()))


def stable_bit_hash(bits: Sequence[int]) -> str:
    """Compact stable hash for a bit list (used in context keys)."""
    payload = ",".join(str(int(b)) for b in bits).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def size_bin_from_heavy_atoms(n_heavy: int) -> str:
    """Coarse size bin to reduce confounding by size regime."""
    n = int(n_heavy)
    if n < 15:
        return "<15"
    if n < 25:
        return "15-24"
    if n < 35:
        return "25-34"
    if n < 50:
        return "35-49"
    return "50+"


@dataclass(frozen=True)
class ContextKey:
    operator: str
    template: str
    site: Tuple[int, ...]
    radius: int
    n_bits: int
    bit_hash: str
    size_bin: str

    def as_str(self) -> str:
        site_s = "(" + ",".join(str(int(x)) for x in self.site) + ")"
        return (
            f"op={self.operator}|tpl={self.template}|site={site_s}|"
            f"r={int(self.radius)}|nb={int(self.n_bits)}|h={self.bit_hash}|sz={self.size_bin}"
        )


def make_context_key(
    *,
    smiles_before: str,
    site: Sequence[int] | Tuple[int, ...],
    operator: str,
    template: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> ContextKey | None:
    """Build a deterministic ContextKey from a single logged step."""
    mol = _ensure_mol(smiles_before)
    if mol is None:
        return None

    site_t = tuple(int(x) for x in site) if site is not None else tuple()
    neighborhood = atoms_within_radius(mol, site_t, int(radius))
    bits = focused_morgan_bits(
        mol, from_atoms=neighborhood, radius=int(radius), n_bits=int(n_bits)
    )
    h = stable_bit_hash(bits)
    size_bin = size_bin_from_heavy_atoms(mol.GetNumHeavyAtoms())

    return ContextKey(
        operator=str(operator),
        template=str(template or ""),
        site=site_t,
        radius=int(radius),
        n_bits=int(n_bits),
        bit_hash=h,
        size_bin=size_bin,
    )
