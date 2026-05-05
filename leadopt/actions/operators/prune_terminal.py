from __future__ import annotations

from typing import List, Sequence

from rdkit import Chem

from ...constraints.base import ConstraintContext
from ...core.errors import ActionError
from ...core.rdkit_utils import assert_valid_mol
from ..base import ActionInstance, ActionOperator, AppliedAction


class PruneTerminal(ActionOperator):
    """
    Remove a single terminal heavy atom (degree==1), not in the locked core.
    Conservative: removes only the terminal atom (not whole fragments).
    """

    name = "PruneTerminal"

    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:
        actions: List[ActionInstance] = []
        for a in mol.GetAtoms():
            i = a.GetIdx()
            if ctx.locked_atoms[i]:
                continue
            if a.GetAtomicNum() == 1:
                continue
            if a.GetDegree() != 1:
                continue

            # allow pruning terminal atoms attached to core or non-core
            nbr = a.GetNeighbors()[0]
            nbr_idx = nbr.GetIdx()

            actions.append(
                ActionInstance(
                    operator=self.name,
                    site=(i,),
                    template="prune_atom",
                    payload={
                        "removed_atom_idx": i,
                        "removed_z": a.GetAtomicNum(),
                        "neighbor_idx": nbr_idx,
                        "neighbor_z": nbr.GetAtomicNum(),
                    },
                )
            )
        # deterministic order
        actions.sort(key=lambda a: a.stable_sort_key())
        return actions

    def touched(
        self, mol: Chem.Mol, action: ActionInstance
    ) -> tuple[set[int], set[int]]:
        atom_idx = int(action.site[0])

        # Conservative: include the terminal atom and the bond that will be removed.
        touched_atoms: set[int] = {atom_idx}
        touched_bonds: set[int] = set()

        if 0 <= atom_idx < mol.GetNumAtoms():
            atom = mol.GetAtomWithIdx(atom_idx)
            # Terminal pruning should involve exactly one bond, but be robust.
            for b in atom.GetBonds():
                touched_bonds.add(int(b.GetIdx()))
                # Also include the neighbor atom (helps constraint locking correctness).
                nbr = b.GetOtherAtomIdx(atom_idx)
                if 0 <= int(nbr) < mol.GetNumAtoms():
                    touched_atoms.add(int(nbr))

        return touched_atoms, touched_bonds

    def apply(self, mol: Chem.Mol, action: ActionInstance) -> AppliedAction:
        self._ensure_operator_match(action)
        atom_idx = int(action.site[0])

        if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
            raise ActionError("Invalid atom index.")

        a = mol.GetAtomWithIdx(atom_idx)
        if a.GetAtomicNum() == 1 or a.GetDegree() != 1:
            raise ActionError("Not a terminal heavy atom.")

        rw = Chem.RWMol(mol)

        # remove the bond first (track which bond index was removed in the original mol)
        nbr = a.GetNeighbors()[0]
        nbr_idx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(atom_idx, nbr_idx)
        if bond is None:
            raise ActionError("Terminal atom has no bond (unexpected).")
        # removed_bond_idx = bond.GetIdx()

        rw.RemoveBond(atom_idx, nbr_idx)
        rw.RemoveAtom(atom_idx)

        new_mol = rw.GetMol()
        assert_valid_mol(new_mol)

        touched_atoms, touched_bonds = self.touched(mol, action)
        # touched() is defined in the ORIGINAL mol index space (as required).
        # Use it verbatim to keep the contract consistent.
        return AppliedAction(
            mol=new_mol,
            action=action,
            touched_atoms=touched_atoms,
            touched_bonds=touched_bonds,
        )
