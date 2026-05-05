from __future__ import annotations

from typing import List, Sequence

from rdkit import Chem

from ...constraints.base import ConstraintContext
from ...core.errors import ActionError
from ...core.rdkit_utils import assert_valid_mol
from ..base import ActionInstance, ActionOperator, AppliedAction


class AzaScanAromatic(ActionOperator):
    """
    Aromatic aza-scan: mutate aromatic carbon -> aromatic nitrogen (C(aryl) -> N(aryl)).

    Conservative v1:
      - only aromatic atoms
      - only atomic num 6 -> 7
      - skips locked atoms (core-safe)
      - no attempt to manage charges; rely on RDKit sanitize + later rule layer

    Deterministic ordering: by atom idx
    """

    name = "AzaScanAromatic"

    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:
        actions: List[ActionInstance] = []

        for a in mol.GetAtoms():
            i = int(a.GetIdx())
            if bool(ctx.locked_atoms[i]):
                continue
            if a.GetAtomicNum() != 6:
                continue
            if not a.GetIsAromatic():
                continue
            if a.GetFormalCharge() != 0:
                continue

            actions.append(
                ActionInstance(
                    operator=self.name,
                    site=(i,),
                    template="aryl_C_to_N",
                    payload={"from": 6, "to": 7},
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

        if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
            raise ActionError("Invalid atom index.")

        rw = Chem.RWMol(mol)
        a = rw.GetAtomWithIdx(atom_idx)

        if a.GetAtomicNum() != 6 or (not a.GetIsAromatic()):
            raise ActionError("Not an aromatic carbon.")
        if a.GetFormalCharge() != 0:
            raise ActionError("Charged aromatic atoms not supported in v1 aza-scan.")

        a.SetAtomicNum(7)  # aromatic N
        new_mol = rw.GetMol()
        assert_valid_mol(new_mol)

        touched_atoms, touched_bonds = self.touched(mol, action)
        return AppliedAction(
            mol=new_mol, action=action, touched_atoms=touched_atoms, touched_bonds=set()
        )
