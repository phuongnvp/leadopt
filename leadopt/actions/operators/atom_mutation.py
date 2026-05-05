from __future__ import annotations

from typing import List, Sequence

from rdkit import Chem

from ...constraints.base import ConstraintContext
from ...core.errors import ActionError
from ...core.rdkit_utils import assert_valid_mol
from ..base import ActionInstance, ActionOperator, AppliedAction


class AtomMutation(ActionOperator):
    """
    Conservative atom mutations to avoid RDKit valence crashes.
    - Halogen swaps (F/Cl/Br/I)
    - Aliphatic C <-> N only when valence-safe (neutral N valence <= 3)
    """

    name = "AtomMutation"

    HALOGENS = (9, 17, 35, 53)  # F, Cl, Br, I

    @staticmethod
    def _is_safe_neutral_n_target(atom: Chem.Atom) -> bool:
        """
        Neutral N cannot exceed typical valence 3 (in RDKit sanitize rules).
        We conservatively allow C->N only if current total valence <= 3.
        """
        # TotalValence counts explicit valence (bond order sum + explicit Hs)
        try:
            tv = int(atom.GetTotalValence())
        except Exception:
            tv = int(atom.GetDegree())
        return tv <= 3 and atom.GetFormalCharge() == 0 and (not atom.GetIsAromatic())

    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:
        actions: List[ActionInstance] = []

        for atom in mol.GetAtoms():
            i = atom.GetIdx()
            if ctx.locked_atoms[i]:
                continue
            if atom.GetAtomicNum() == 1:
                continue

            z = atom.GetAtomicNum()
            is_arom = atom.GetIsAromatic()

            # Halogen swaps
            if z in self.HALOGENS:
                for z2 in self.HALOGENS:
                    if z2 == z:
                        continue
                    actions.append(
                        ActionInstance(
                            operator=self.name,
                            site=(i,),
                            template=f"{z}->{z2}",
                            payload={"from": z, "to": z2},
                        )
                    )

            # Conservative C<->N only for non-aromatic atoms
            if not is_arom and z in (6, 7):
                if z == 6:
                    # Only allow C->N if resulting neutral N valence likely <= 3
                    if self._is_safe_neutral_n_target(atom):
                        actions.append(
                            ActionInstance(
                                operator=self.name,
                                site=(i,),
                                template="C->N",
                                payload={"from": 6, "to": 7},
                            )
                        )
                else:
                    # N->C is usually valence-safe, still non-aromatic only
                    actions.append(
                        ActionInstance(
                            operator=self.name,
                            site=(i,),
                            template="N->C",
                            payload={"from": 7, "to": 6},
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
            raise ActionError("Missing mutation payload.")

        to_z = int(payload["to"])

        if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
            raise ActionError("Invalid atom index.")

        rw = Chem.RWMol(mol)
        atom = rw.GetAtomWithIdx(atom_idx)

        # Re-check safety at apply time (important!)
        if to_z == 7:  # targeting N
            if not self._is_safe_neutral_n_target(atom):
                raise ActionError(
                    "Unsafe C->N mutation (would violate neutral N valence)."
                )

        # Keep aromatic restriction
        if atom.GetIsAromatic() and atom.GetAtomicNum() in (6, 7) and to_z in (6, 7):
            raise ActionError("Disallowed aromatic C<->N mutation (conservative).")

        atom.SetAtomicNum(to_z)

        new_mol = rw.GetMol()
        assert_valid_mol(new_mol)

        touched_atoms, touched_bonds = self.touched(mol, action)
        return AppliedAction(
            mol=new_mol, action=action, touched_atoms=touched_atoms, touched_bonds=set()
        )
