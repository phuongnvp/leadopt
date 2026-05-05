from __future__ import annotations

from typing import Tuple

import torch
from rdkit import Chem

# Keep feature space stable and small (extend later)
ATOM_Z = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H,C,N,O,F,P,S,Cl,Br,I
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]


def _one_hot(x, vocab) -> list[int]:
    return [1 if x == v else 0 for v in vocab]


def mol_to_graph_tensors(
    mol: Chem.Mol, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      x: [N, F_atom]
      edge_index: [2, E2] (bidirectional)
      edge_attr: [E2, F_bond]
    """
    n = mol.GetNumAtoms()
    if n == 0:
        raise ValueError("Empty molecule.")

    atom_feats = []
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        deg = a.GetDegree()
        arom = int(a.GetIsAromatic())
        charge = a.GetFormalCharge()

        # atom feature vector
        feats = []
        feats += _one_hot(z, ATOM_Z)
        feats += [min(deg, 6) / 6.0]  # normalized-ish
        feats += [arom]
        feats += [max(-3, min(charge, 3)) / 3.0]  # rough normalization
        atom_feats.append(feats)

    x = torch.tensor(atom_feats, dtype=torch.float32, device=device)

    # bonds -> edges (bidirectional)
    rows = []
    cols = []
    bond_feats = []

    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bt = b.GetBondType()
        arom = int(b.GetIsAromatic())

        bf = []
        bf += _one_hot(bt, BOND_TYPES)
        bf += [arom]
        bf = torch.tensor(bf, dtype=torch.float32, device=device)

        # i->j and j->i
        rows += [i, j]
        cols += [j, i]
        bond_feats += [bf, bf]

    edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
    edge_attr = (
        torch.stack(bond_feats, dim=0)
        if bond_feats
        else torch.zeros((0, len(BOND_TYPES) + 1), device=device)
    )

    # batch = torch.zeros((n,), dtype=torch.long, device=device)  # single-graph batch id
    return x, edge_index, edge_attr
