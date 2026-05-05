from __future__ import annotations

from typing import List, Sequence

from rdkit import Chem

from ...constraints.base import ConstraintContext
from ...core.errors import ActionError
from ...core.rdkit_utils import assert_valid_mol
from ..base import ActionInstance, ActionOperator, AppliedAction


class FunctionalGroupSwap(ActionOperator):
    """
    Swap terminal functional groups in a very conservative way:
      - terminal OH (O, degree 1, has H) <-> terminal NH2 (N, degree 1)
    Only on non-core atoms.

    This is intentionally limited; you can expand using SMARTS reactions later.
    """

    name = "FunctionalGroupSwap"

    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:
        actions: List[ActionInstance] = []

        for atom in mol.GetAtoms():
            i = atom.GetIdx()
            if ctx.locked_atoms[i]:
                continue
            if atom.GetDegree() != 1:
                continue

            z = atom.GetAtomicNum()

            # terminal alcohol oxygen
            if z == 8 and atom.GetTotalNumHs() >= 1:
                actions.append(
                    ActionInstance(
                        operator=self.name,
                        site=(i,),
                        template="OH_to_NH2",
                        payload={"from": 8, "to": 7},
                    )
                )

            # terminal amine nitrogen (degree 1)
            if z == 7:
                actions.append(
                    ActionInstance(
                        operator=self.name,
                        site=(i,),
                        template="NH2_to_OH",
                        payload={"from": 7, "to": 8},
                    )
                )

        actions.sort(key=lambda a: a.stable_sort_key())
        return actions

    def touched(
        self, mol: Chem.Mol, action: ActionInstance
    ) -> tuple[set[int], set[int]]:
        self._ensure_operator_match(action)
        return {int(action.site[0])}, set()

    def apply(self, mol: Chem.Mol, action: ActionInstance) -> AppliedAction:
        self._ensure_operator_match(action)
        atom_idx = int(action.site[0])
        payload = action.payload or {}
        if "to" not in payload:
            raise ActionError("Missing swap payload.")
        to_z = int(payload["to"])

        if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
            raise ActionError("Invalid atom index.")

        rw = Chem.RWMol(mol)
        atom = rw.GetAtomWithIdx(atom_idx)

        if atom.GetDegree() != 1:
            raise ActionError("Not a terminal atom.")

        atom.SetAtomicNum(to_z)

        new_mol = rw.GetMol()
        assert_valid_mol(new_mol)

        touched_atoms, touched_bonds = self.touched(mol, action)
        return AppliedAction(
            mol=new_mol, action=action, touched_atoms=touched_atoms, touched_bonds=set()
        )
