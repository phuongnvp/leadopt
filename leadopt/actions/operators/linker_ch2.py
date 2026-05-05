from __future__ import annotations

from typing import List, Sequence

from rdkit import Chem

from ...constraints.base import ConstraintContext
from ...core.errors import ActionError
from ...core.rdkit_utils import assert_valid_mol
from ..base import ActionInstance, ActionOperator, AppliedAction


def _bond_key(b: Chem.Bond) -> tuple[int, int]:
    a = b.GetBeginAtomIdx()
    c = b.GetEndAtomIdx()
    return (a, c) if a <= c else (c, a)


class LinkerInsertCH2(ActionOperator):
    """
    Insert a methylene (-CH2-) into an acyclic, non-aromatic single bond A-B:

        A—B  ->  A—CH2—B

    Conservative v1:
      - only SINGLE bonds
      - bond not aromatic
      - bond not in ring
      - both atoms heavy (non-H)
      - do NOT touch locked bonds/atoms directly (masking via constraint system handles core)
    """

    name = "LinkerInsertCH2"

    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:
        actions: List[ActionInstance] = []

        locked_bonds = getattr(ctx, "locked_bonds", None)

        for b in mol.GetBonds():
            bidx = b.GetIdx()

            if locked_bonds is not None and bool(locked_bonds[bidx]):
                continue

            if b.GetBondType() != Chem.BondType.SINGLE:
                continue
            if b.GetIsAromatic():
                continue
            if b.IsInRing():
                continue

            a1 = b.GetBeginAtom()
            a2 = b.GetEndAtom()

            if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
                continue

            # Optional conservative exclusions:
            # - avoid inserting into bonds adjacent to dummy atoms (shouldn't exist)
            if a1.GetAtomicNum() == 0 or a2.GetAtomicNum() == 0:
                continue

            # Deterministic site uses (min_idx, max_idx, bond_idx)
            i = int(a1.GetIdx())
            j = int(a2.GetIdx())
            ii, jj = (i, j) if i <= j else (j, i)

            actions.append(
                ActionInstance(
                    operator=self.name,
                    site=(ii, jj, int(bidx)),
                    template="insert_CH2",
                    payload={
                        "a1": ii,
                        "a2": jj,
                        "bond_idx": int(bidx),
                    },
                )
            )

        actions.sort(key=lambda a: a.stable_sort_key())
        return actions

    def touched(
        self, mol: Chem.Mol, action: ActionInstance
    ) -> tuple[set[int], set[int]]:
        self._ensure_operator_match(action)
        payload = action.payload or {}
        a1 = int(payload.get("a1", action.site[0]))
        a2 = int(payload.get("a2", action.site[1]))
        bidx = int(payload.get("bond_idx", action.site[2]))
        # touches both endpoints + the bond being modified
        return {a1, a2}, {bidx}

    def apply(self, mol: Chem.Mol, action: ActionInstance) -> AppliedAction:
        self._ensure_operator_match(action)
        payload = action.payload or {}
        if "a1" not in payload or "a2" not in payload:
            raise ActionError("Missing payload keys for LinkerInsertCH2.")
        a1 = int(payload["a1"])
        a2 = int(payload["a2"])

        if a1 < 0 or a1 >= mol.GetNumAtoms() or a2 < 0 or a2 >= mol.GetNumAtoms():
            raise ActionError("Invalid atom indices.")
        if a1 == a2:
            raise ActionError("Degenerate bond endpoints.")

        b = mol.GetBondBetweenAtoms(a1, a2)
        if b is None:
            raise ActionError("No bond between specified atoms.")
        if b.GetBondType() != Chem.BondType.SINGLE or b.GetIsAromatic() or b.IsInRing():
            raise ActionError("Bond not eligible for CH2 insertion.")

        rw = Chem.RWMol(mol)

        # Remove original bond
        rw.RemoveBond(a1, a2)

        # Add new carbon (atomic number 6)
        c_idx = rw.AddAtom(Chem.Atom(6))

        # Add A1—C—A2 bonds
        rw.AddBond(a1, c_idx, Chem.BondType.SINGLE)
        rw.AddBond(c_idx, a2, Chem.BondType.SINGLE)

        new_mol = rw.GetMol()
        assert_valid_mol(new_mol)

        touched_atoms, touched_bonds = self.touched(mol, action)
        # bond indices change after edit; return original bond idx conservatively
        # return AppliedAction(mol=new_mol, action=action, touched_atoms=touched_atoms | {int(c_idx)}, touched_bonds=touched_bonds)
        return AppliedAction(
            mol=new_mol,
            action=action,
            touched_atoms=touched_atoms,
            touched_bonds=touched_bonds,
        )


class LinkerDeleteCH2(ActionOperator):
    """
    Delete a methylene (-CH2-) linker carbon that is:
      - atomic num = 6
      - degree == 2
      - not aromatic
      - not in ring
    and connect its two neighbors A and B:

        A—CH2—B  ->  A—B

    Conservative v1:
      - only if A—B bond does not already exist
      - only if the two bonds are single and non-aromatic
    """

    name = "LinkerDeleteCH2"

    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:
        actions: List[ActionInstance] = []

        for a in mol.GetAtoms():
            i = int(a.GetIdx())

            # Don't delete locked atoms (important for core)
            if bool(ctx.locked_atoms[i]):
                continue

            if a.GetAtomicNum() != 6:
                continue
            if a.GetIsAromatic():
                continue
            if a.IsInRing():
                continue
            if a.GetDegree() != 2:
                continue

            nbs = list(a.GetNeighbors())
            if len(nbs) != 2:
                continue

            n1 = int(nbs[0].GetIdx())
            n2 = int(nbs[1].GetIdx())
            if n1 == n2:
                continue

            # both adjacent bonds must be single, non-aromatic
            b1 = mol.GetBondBetweenAtoms(i, n1)
            b2 = mol.GetBondBetweenAtoms(i, n2)
            if b1 is None or b2 is None:
                continue
            if (
                b1.GetBondType() != Chem.BondType.SINGLE
                or b2.GetBondType() != Chem.BondType.SINGLE
            ):
                continue
            if b1.GetIsAromatic() or b2.GetIsAromatic():
                continue

            # do not create duplicate bond
            if mol.GetBondBetweenAtoms(n1, n2) is not None:
                continue

            # deterministic site = (carbon_idx, min(n1,n2), max(n1,n2))
            nn1, nn2 = (n1, n2) if n1 <= n2 else (n2, n1)
            actions.append(
                ActionInstance(
                    operator=self.name,
                    site=(i, nn1, nn2),
                    template="delete_CH2",
                    payload={
                        "c": i,
                        "n1": nn1,
                        "n2": nn2,
                    },
                )
            )

        actions.sort(key=lambda a: a.stable_sort_key())
        return actions

    def touched(
        self, mol: Chem.Mol, action: ActionInstance
    ) -> tuple[set[int], set[int]]:
        self._ensure_operator_match(action)
        payload = action.payload or {}
        c = int(payload.get("c", action.site[0]))
        n1 = int(payload.get("n1", action.site[1]))
        n2 = int(payload.get("n2", action.site[2]))

        b1 = mol.GetBondBetweenAtoms(c, n1)
        b2 = mol.GetBondBetweenAtoms(c, n2)
        touched_bonds: set[int] = set()
        if b1 is not None:
            touched_bonds.add(int(b1.GetIdx()))
        if b2 is not None:
            touched_bonds.add(int(b2.GetIdx()))

        return {c, n1, n2}, touched_bonds

    def apply(self, mol: Chem.Mol, action: ActionInstance) -> AppliedAction:
        self._ensure_operator_match(action)
        payload = action.payload or {}
        if "c" not in payload or "n1" not in payload or "n2" not in payload:
            raise ActionError("Missing payload keys for LinkerDeleteCH2.")
        c = int(payload["c"])
        n1 = int(payload["n1"])
        n2 = int(payload["n2"])

        if any(x < 0 or x >= mol.GetNumAtoms() for x in (c, n1, n2)):
            raise ActionError("Invalid atom indices.")
        if c == n1 or c == n2 or n1 == n2:
            raise ActionError("Degenerate indices.")

        atom = mol.GetAtomWithIdx(c)
        if (
            atom.GetAtomicNum() != 6
            or atom.GetIsAromatic()
            or atom.IsInRing()
            or atom.GetDegree() != 2
        ):
            raise ActionError("Atom is not an eligible CH2 linker carbon.")

        # must have two single bonds to n1/n2
        b1 = mol.GetBondBetweenAtoms(c, n1)
        b2 = mol.GetBondBetweenAtoms(c, n2)
        if b1 is None or b2 is None:
            raise ActionError("Missing expected bonds to neighbors.")
        if (
            b1.GetBondType() != Chem.BondType.SINGLE
            or b2.GetBondType() != Chem.BondType.SINGLE
        ):
            raise ActionError("Non-single bond adjacent to linker carbon.")
        if b1.GetIsAromatic() or b2.GetIsAromatic():
            raise ActionError("Aromatic bond adjacent to linker carbon.")

        # avoid duplicate bond
        if mol.GetBondBetweenAtoms(n1, n2) is not None:
            raise ActionError("Neighbors already bonded; would create duplicate bond.")

        rw = Chem.RWMol(mol)

        # Add new bond first, then remove carbon
        rw.AddBond(n1, n2, Chem.BondType.SINGLE)

        # Remove bonds from c (safe even if indices shift after first removal)
        rw.RemoveBond(c, n1)
        rw.RemoveBond(c, n2)
        rw.RemoveAtom(c)

        new_mol = rw.GetMol()
        assert_valid_mol(new_mol)

        touched_atoms, touched_bonds = self.touched(mol, action)
        return AppliedAction(
            mol=new_mol,
            action=action,
            touched_atoms=touched_atoms,
            touched_bonds=touched_bonds,
        )
