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


class DeleteSubtree(ActionOperator):
    """
    Delete a side-fragment by cutting a single (non-ring, non-aromatic) bond and removing one side.

    Intended for "decomplexification":
      - works even when the removable fragment is NOT terminal
      - respects core constraints: never deletes locked atoms
      - deterministic: for unconstrained case, deletes the smaller side (ties broken deterministically)

    Notes:
      - Only considers SINGLE bonds that are NOT in a ring and NOT aromatic.
      - Only deletes heavy-atom fragments (won’t consider bonds to H, but those usually don’t exist explicitly).
    """

    name = "DeleteSubtree"

    def __init__(self, *, max_deleted_atoms: int = 25) -> None:
        self.max_deleted_atoms = int(max_deleted_atoms)

    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:
        actions: List[ActionInstance] = []

        n_atoms = mol.GetNumAtoms()
        if n_atoms < 2:
            return actions

        for bond in mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            if bond.GetIsAromatic():
                continue
            if bond.IsInRing():
                continue

            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()

            # avoid deleting across bonds involving dummy atoms or H (very conservative)
            if mol.GetAtomWithIdx(a).GetAtomicNum() <= 1:
                continue
            if mol.GetAtomWithIdx(b).GetAtomicNum() <= 1:
                continue

            comp_a = _component_atoms_excluding_bond(mol, a, (a, b))
            comp_b = _component_atoms_excluding_bond(mol, b, (a, b))

            # Both components must cover all atoms (sanity)
            if len(comp_a) + len(comp_b) != n_atoms:
                # This can happen if the graph has oddities; skip conservatively
                continue

            locked = ctx.locked_atoms

            a_has_locked = any(locked[i] for i in comp_a)
            b_has_locked = any(locked[i] for i in comp_b)

            # If both sides contain locked atoms, cannot delete either side safely
            if a_has_locked and b_has_locked:
                continue

            # Determine which side to delete
            if a_has_locked and not b_has_locked:
                delete_set = comp_b
                keep_root = a
            elif b_has_locked and not a_has_locked:
                delete_set = comp_a
                keep_root = b
            else:
                # Unconstrained (or no locked atoms on either side):
                # delete smaller component; ties broken deterministically
                if len(comp_a) < len(comp_b):
                    delete_set = comp_a
                    keep_root = b
                elif len(comp_b) < len(comp_a):
                    delete_set = comp_b
                    keep_root = a
                else:
                    # tie-breaker: delete the one with larger max atom index (deterministic)
                    if max(comp_a) > max(comp_b):
                        delete_set = comp_a
                        keep_root = b
                    else:
                        delete_set = comp_b
                        keep_root = a

            # Don’t allow deleting entire molecule
            if len(delete_set) >= n_atoms:
                continue

            # Respect size cap (prevents huge destructive deletions)
            if len(delete_set) > self.max_deleted_atoms:
                continue

            # Avoid deleting locked atoms (should already be handled)
            if any(locked[i] for i in delete_set):
                continue

            actions.append(
                ActionInstance(
                    operator=self.name,
                    site=(int(a), int(b)),  # bond endpoints
                    template="delete_subtree",
                    payload={
                        "a": int(a),
                        "b": int(b),
                        "bond_idx": int(bond.GetIdx()),
                        "keep_root": int(keep_root),
                        "delete_atoms": sorted(int(i) for i in delete_set),
                        "n_deleted": int(len(delete_set)),
                    },
                )
            )

        # Deterministic ordering
        actions.sort(key=lambda a: a.stable_sort_key())
        return actions

    def touched(
        self, mol: Chem.Mol, action: ActionInstance
    ) -> tuple[set[int], set[int]]:
        payload = action.payload or {}
        touched_atoms: set[int] = set()
        touched_bonds: set[int] = set()

        for i in payload.get("delete_atoms", []) or []:
            j = int(i)
            if 0 <= j < mol.GetNumAtoms():
                touched_atoms.add(j)

        keep_root = payload.get("keep_root", None)
        if keep_root is not None:
            j = int(keep_root)
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

        if "a" not in payload or "b" not in payload or "delete_atoms" not in payload:
            raise ActionError("DeleteSubtree missing payload fields.")

        a = int(payload["a"])
        b = int(payload["b"])
        delete_atoms = [int(i) for i in payload["delete_atoms"]]
        delete_set = set(delete_atoms)

        nA = mol.GetNumAtoms()
        if a < 0 or a >= nA or b < 0 or b >= nA:
            raise ActionError("Invalid bond endpoints.")

        # Bond must exist in the current molecule
        if mol.GetBondBetweenAtoms(a, b) is None:
            raise ActionError(
                "Bond between endpoints not found (molecule changed unexpectedly)."
            )

        # Validate keep_root consistency (enumerate_actions sets this)
        keep_root = payload.get("keep_root", None)
        if keep_root is None:
            raise ActionError("DeleteSubtree missing keep_root in payload.")
        keep_root = int(keep_root)

        if keep_root not in (a, b):
            raise ActionError("keep_root must be one of the bond endpoints (a or b).")

        if keep_root in delete_set:
            raise ActionError(
                "keep_root lies in delete_atoms (inconsistent action payload)."
            )

        # Endpoint sanity: never allow deleting both endpoints
        if (a in delete_set) and (b in delete_set):
            raise ActionError(
                "Both bond endpoints are in delete_atoms (inconsistent action payload)."
            )

        # Remove atoms on the delete side (descending order so indices remain valid)
        rw = Chem.RWMol(mol)
        for idx in sorted(delete_atoms, reverse=True):
            if idx < 0 or idx >= rw.GetNumAtoms():
                raise ActionError("delete_atoms contains invalid indices.")
            rw.RemoveAtom(idx)

        new_mol = rw.GetMol()

        # If deletion produced multiple fragments, keep the fragment containing keep_root.
        # Map keep_root (old index) -> new index after deletions:
        n_deleted_before = sum(1 for x in delete_atoms if x < keep_root)
        keep_new = keep_root - n_deleted_before
        if keep_new < 0 or keep_new >= new_mol.GetNumAtoms():
            raise ActionError("Computed keep_root index out of range after deletion.")

        frags_idx = Chem.GetMolFrags(new_mol, asMols=False, sanitizeFrags=False)
        if len(frags_idx) > 1:
            keep_frag_i = None
            for i, frag in enumerate(frags_idx):
                if keep_new in frag:
                    keep_frag_i = i
                    break
            if keep_frag_i is None:
                raise ActionError(
                    "Could not find kept fragment after deletion (unexpected)."
                )

            frags_mol = Chem.GetMolFrags(new_mol, asMols=True, sanitizeFrags=True)
            new_mol = frags_mol[int(keep_frag_i)]

        assert_valid_mol(new_mol)

        touched_atoms, touched_bonds = self.touched(mol, action)
        return AppliedAction(
            mol=new_mol,
            action=action,
            touched_atoms=touched_atoms,
            touched_bonds=touched_bonds,
        )
