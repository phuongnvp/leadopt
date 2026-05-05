from __future__ import annotations

from collections import deque
from typing import List, Sequence, Set, Tuple

from rdkit import Chem

from ...constraints.base import ConstraintContext
from ...core.errors import ActionError
from ...core.rdkit_utils import assert_valid_mol
from ..base import ActionInstance, ActionOperator, AppliedAction


def _component_atoms_excluding_bond(
    mol: Chem.Mol, start: int, blocked: Tuple[int, int]
) -> Set[int]:
    """BFS to collect atoms reachable from start without traversing the blocked bond (u,v)."""
    u, v = blocked
    seen: Set[int] = set()
    q = deque([start])
    seen.add(start)

    while q:
        x = q.popleft()
        ax = mol.GetAtomWithIdx(x)
        for nb in ax.GetNeighbors():
            y = nb.GetIdx()
            # block traversal across the selected bond in either direction
            if (x == u and y == v) or (x == v and y == u):
                continue
            if y not in seen:
                seen.add(y)
                q.append(y)
    return seen


class RingSubstituentDelete(ActionOperator):
    """Delete non-ring substituents attached to ring atoms while preserving the ring core.

    This operator is intended for medchem-style pruning of ring-attached sidechains.

    Hard rule:
      - never cleaves ring bonds (only deletes a non-ring subtree attached via a single bond).

    Determinism:
      - enumeration is sorted by ActionInstance.stable_sort_key().

    Notes:
      - Only considers SINGLE, non-aromatic bonds that are NOT in a ring.
      - Only considers cases where exactly one bond endpoint is in a ring.
      - Respects locked atoms: will not delete any locked atoms.
    """

    name = "RingSubstituentDelete"

    def __init__(
        self,
        *,
        max_deleted_atoms: int = 25,
        min_heavy_atoms: int = 5,
        max_fragments: int = 1,
    ) -> None:
        self.max_deleted_atoms = int(max_deleted_atoms)
        self.min_heavy_atoms = int(min_heavy_atoms)
        self.max_fragments = int(max_fragments)

    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:
        actions: List[ActionInstance] = []

        n_atoms = mol.GetNumAtoms()
        if n_atoms < 2:
            return actions

        locked = ctx.locked_atoms

        for bond in mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            if bond.GetIsAromatic():
                continue
            if bond.IsInRing():
                continue

            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()

            atom_a = mol.GetAtomWithIdx(a)
            atom_b = mol.GetAtomWithIdx(b)

            # avoid dummy atoms or H
            if atom_a.GetAtomicNum() <= 1 or atom_b.GetAtomicNum() <= 1:
                continue

            a_in_ring = atom_a.IsInRing()
            b_in_ring = atom_b.IsInRing()

            # Require exactly one endpoint to be in a ring.
            if a_in_ring == b_in_ring:
                continue

            ring_atom = a if a_in_ring else b
            sub_atom = b if a_in_ring else a

            # Delete the component on the non-ring side.
            delete_set = _component_atoms_excluding_bond(
                mol, sub_atom, (ring_atom, sub_atom)
            )

            # Never delete the ring atom.
            if ring_atom in delete_set:
                continue

            # Size caps / sanity
            if len(delete_set) == 0 or len(delete_set) >= n_atoms:
                continue
            if len(delete_set) > self.max_deleted_atoms:
                continue

            # Respect locked atoms
            if any(locked[i] for i in delete_set):
                continue

            # Minimum size after deletion
            if (n_atoms - len(delete_set)) < self.min_heavy_atoms:
                continue

            actions.append(
                ActionInstance(
                    operator=self.name,
                    site=(int(ring_atom), int(sub_atom)),
                    template="ring_substituent_delete",
                    payload={
                        "ring_atom": int(ring_atom),
                        "sub_atom": int(sub_atom),
                        "bond_idx": int(bond.GetIdx()),
                        "delete_atoms": sorted(int(i) for i in delete_set),
                        "n_deleted": int(len(delete_set)),
                        "min_heavy_atoms": int(self.min_heavy_atoms),
                        "max_fragments": int(self.max_fragments),
                    },
                )
            )

        actions.sort(key=lambda a: a.stable_sort_key())
        return actions

    def touched(
        self, mol: Chem.Mol, action: ActionInstance
    ) -> tuple[set[int], set[int]]:
        payload = action.payload or {}
        touched_atoms: set[int] = set()
        touched_bonds: set[int] = set()

        ring_atom = payload.get("ring_atom", None)
        if ring_atom is not None:
            j = int(ring_atom)
            if 0 <= j < mol.GetNumAtoms():
                touched_atoms.add(j)

        for i in payload.get("delete_atoms", []) or []:
            j = int(i)
            if 0 <= j < mol.GetNumAtoms():
                touched_atoms.add(j)

        bond_idx = payload.get("bond_idx", None)
        if bond_idx is not None:
            b = int(bond_idx)
            if 0 <= b < mol.GetNumBonds():
                touched_bonds.add(b)

        return touched_atoms, touched_bonds

    def apply(self, mol: Chem.Mol, action: ActionInstance) -> AppliedAction:
        self._ensure_operator_match(action)
        payload = action.payload or {}

        required = {
            "ring_atom",
            "sub_atom",
            "delete_atoms",
            "min_heavy_atoms",
            "max_fragments",
        }
        if not required.issubset(payload.keys()):
            raise ActionError("RingSubstituentDelete missing payload fields.")

        ring_atom = int(payload["ring_atom"])
        sub_atom = int(payload["sub_atom"])
        delete_atoms = [int(i) for i in payload["delete_atoms"]]
        delete_set = set(delete_atoms)
        min_heavy_atoms = int(payload["min_heavy_atoms"])
        max_fragments = int(payload["max_fragments"])

        nA = mol.GetNumAtoms()
        if ring_atom < 0 or ring_atom >= nA or sub_atom < 0 or sub_atom >= nA:
            raise ActionError("Invalid atom indices.")

        if ring_atom in delete_set:
            raise ActionError("ring_atom lies in delete_atoms (inconsistent payload).")

        # Bond must exist
        if mol.GetBondBetweenAtoms(ring_atom, sub_atom) is None:
            raise ActionError(
                "Bond between ring_atom and sub_atom not found (molecule changed unexpectedly)."
            )

        # Remove atoms on the delete side (descending order so indices remain valid)
        rw = Chem.RWMol(mol)
        for idx in sorted(delete_atoms, reverse=True):
            if idx < 0 or idx >= rw.GetNumAtoms():
                raise ActionError("delete_atoms contains invalid indices.")
            rw.RemoveAtom(idx)

        new_mol = rw.GetMol()

        # Identify the fragment containing the (updated) ring atom index.
        n_deleted_before = sum(1 for x in delete_atoms if x < ring_atom)
        ring_new = ring_atom - n_deleted_before
        if ring_new < 0 or ring_new >= new_mol.GetNumAtoms():
            raise ActionError("Computed ring atom index out of range after deletion.")

        frags_idx = Chem.GetMolFrags(new_mol, asMols=False, sanitizeFrags=False)
        if len(frags_idx) > 1:
            keep_frag_i = None
            for i, frag in enumerate(frags_idx):
                if ring_new in frag:
                    keep_frag_i = i
                    break
            if keep_frag_i is None:
                raise ActionError(
                    "Could not find kept fragment after deletion (unexpected)."
                )

            frags_mol = Chem.GetMolFrags(new_mol, asMols=True, sanitizeFrags=True)
            new_mol = frags_mol[int(keep_frag_i)]

        # Fragment cap
        n_frags_final = len(
            Chem.GetMolFrags(new_mol, asMols=False, sanitizeFrags=False)
        )
        if n_frags_final > max_fragments:
            raise ActionError(
                f"Deletion produced {n_frags_final} fragments (> {max_fragments})."
            )

        if new_mol.GetNumAtoms() < min_heavy_atoms:
            raise ActionError("Result below min_heavy_atoms.")

        assert_valid_mol(new_mol)

        touched_atoms, touched_bonds = self.touched(mol, action)
        return AppliedAction(
            mol=new_mol,
            action=action,
            touched_atoms=touched_atoms,
            touched_bonds=touched_bonds,
        )
