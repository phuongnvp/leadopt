from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

from rdkit import Chem

from ...constraints.base import ConstraintContext
from ...core.errors import ActionError
from ...core.rdkit_utils import assert_valid_mol
from ..base import ActionInstance, ActionOperator, AppliedAction


def _total_formal_charge(mol: Chem.Mol) -> int:
    return int(sum(int(a.GetFormalCharge()) for a in mol.GetAtoms()))


def _is_aromatic_ring(mol: Chem.Mol, ring: Tuple[int, ...]) -> bool:
    # Strict: only 5/6-member rings and all atoms aromatic
    if len(ring) not in (5, 6):
        return False
    for ai in ring:
        a = mol.GetAtomWithIdx(int(ai))
        if not a.GetIsAromatic():
            return False
    return True


def _non_ring_neighbors(atom: Chem.Atom, ring_set: Set[int]) -> List[Chem.Atom]:
    return [n for n in atom.GetNeighbors() if int(n.GetIdx()) not in ring_set]


def _substituent_atoms(mol: Chem.Mol, root_idx: int, ring_set: Set[int]) -> Set[int]:
    """Return the substituent subtree atom indices starting at root_idx, not entering ring_set."""
    out: Set[int] = set()
    frontier: List[int] = [int(root_idx)]
    out.add(int(root_idx))
    while frontier:
        cur = int(frontier.pop())
        a = mol.GetAtomWithIdx(cur)
        for n in a.GetNeighbors():
            ni = int(n.GetIdx())
            if ni in ring_set:
                continue
            if ni not in out:
                out.add(ni)
                frontier.append(ni)
    return out


@dataclass(frozen=True)
class _ScanSite:
    ring_atoms: Tuple[int, ...]
    source_ring_atom: int
    substituent_root: int


class AromaticPositionalScan(ActionOperator):
    """Tier 4.4 — Move a single substituent around an aromatic ring.

    Scope (strict):
      - 5- or 6-member aromatic ring
      - exactly one non-ring substituent attached to the ring (mono-substituted w.r.t. that ring)
      - move that substituent to other ring atoms that are unsubstituted and have an available H

    The substituent identity is preserved exactly (same atoms, reattached).
    """

    name = "AromaticPositionalScan"

    def __init__(
        self,
        *,
        min_heavy_atoms: int = 5,
        allow_charge_change: bool = False,
    ) -> None:
        self.min_heavy_atoms = int(min_heavy_atoms)
        self.allow_charge_change = bool(allow_charge_change)

    def _enumerate_sites(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> List[_ScanSite]:
        ring_info = mol.GetRingInfo()
        rings = list(ring_info.AtomRings())
        # Deterministic ring processing order.
        rings.sort(key=lambda r: tuple(int(i) for i in r))

        sites: List[_ScanSite] = []

        for ring in rings:
            ring_t = tuple(int(i) for i in ring)
            if not _is_aromatic_ring(mol, ring_t):
                continue

            ring_set = set(ring_t)

            # Identify ring atoms with non-ring substituents (heavy atoms only).
            substituted: List[Tuple[int, int]] = (
                []
            )  # (ring_atom_idx, substituent_root_idx)
            ok = True
            for rai in ring_t:
                a = mol.GetAtomWithIdx(int(rai))
                nn = _non_ring_neighbors(a, ring_set)
                nn = [x for x in nn if x.GetAtomicNum() > 1]
                if len(nn) == 0:
                    continue
                if len(nn) != 1:
                    ok = False
                    break
                substituted.append((int(rai), int(nn[0].GetIdx())))

            if not ok:
                continue
            if len(substituted) != 1:
                continue

            src_ring, sub_root = substituted[0]

            # Respect locked atoms: do not use if source ring atom is locked.
            if bool(ctx.locked_atoms.get(int(src_ring), False)):
                continue

            # Ensure the substituent bond is a detachable single, non-aromatic, non-ring bond.
            b = mol.GetBondBetweenAtoms(int(src_ring), int(sub_root))
            if b is None:
                continue
            if b.GetBondType() != Chem.BondType.SINGLE:
                continue
            if b.GetIsAromatic() or b.IsInRing():
                continue

            sites.append(
                _ScanSite(
                    ring_atoms=ring_t,
                    source_ring_atom=int(src_ring),
                    substituent_root=int(sub_root),
                )
            )

        return sites

    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:
        actions: List[ActionInstance] = []
        sites = self._enumerate_sites(mol, ctx)

        for site in sites:
            ring_set = set(int(i) for i in site.ring_atoms)

            # Valid targets: unsubstituted ring atoms with at least one H, not locked.
            for tgt in site.ring_atoms:
                tgt_i = int(tgt)
                if tgt_i == int(site.source_ring_atom):
                    continue

                if bool(ctx.locked_atoms.get(tgt_i, False)):
                    continue

                tgt_atom = mol.GetAtomWithIdx(tgt_i)
                if int(tgt_atom.GetTotalNumHs()) <= 0:
                    continue

                nn = _non_ring_neighbors(tgt_atom, ring_set)
                nn = [x for x in nn if x.GetAtomicNum() > 1]
                if len(nn) != 0:
                    continue

                actions.append(
                    ActionInstance(
                        operator=self.name,
                        site=(int(site.source_ring_atom), int(tgt_i)),
                        template="move",
                        payload={
                            "ring_atoms": [int(i) for i in site.ring_atoms],
                            "source_ring_atom": int(site.source_ring_atom),
                            "target_ring_atom": int(tgt_i),
                            "substituent_root": int(site.substituent_root),
                            "min_heavy_atoms": int(self.min_heavy_atoms),
                            "allow_charge_change": bool(self.allow_charge_change),
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
        src = int(payload.get("source_ring_atom", action.site[0]))
        tgt = int(payload.get("target_ring_atom", action.site[1]))
        root = int(payload.get("substituent_root"))
        ring_atoms = set(int(i) for i in (payload.get("ring_atoms") or []))

        touched_atoms: Set[int] = {int(src), int(tgt)}
        if ring_atoms and 0 <= root < mol.GetNumAtoms():
            touched_atoms |= _substituent_atoms(mol, root, ring_atoms)
        else:
            touched_atoms.add(int(root))

        touched_bonds: Set[int] = set()
        b = mol.GetBondBetweenAtoms(int(src), int(root))
        if b is not None:
            touched_bonds.add(int(b.GetIdx()))

        return set(int(i) for i in touched_atoms), set(int(i) for i in touched_bonds)

    def apply(self, mol: Chem.Mol, action: ActionInstance) -> AppliedAction:
        self._ensure_operator_match(action)
        payload = action.payload or {}

        ring_atoms = tuple(int(i) for i in payload["ring_atoms"])
        ring_set = set(int(i) for i in ring_atoms)

        src = int(payload["source_ring_atom"])
        tgt = int(payload["target_ring_atom"])
        root = int(payload["substituent_root"])

        before_charge = _total_formal_charge(mol)

        if (
            src < 0
            or src >= mol.GetNumAtoms()
            or tgt < 0
            or tgt >= mol.GetNumAtoms()
            or root < 0
            or root >= mol.GetNumAtoms()
        ):
            raise ActionError("Invalid atom index in AromaticPositionalScan payload.")
        if src not in ring_set or tgt not in ring_set:
            raise ActionError("Source/target not in ring_atoms.")
        if root in ring_set:
            raise ActionError("Substituent root must be non-ring.")

        b = mol.GetBondBetweenAtoms(int(src), int(root))
        if b is None:
            raise ActionError("No bond between source ring atom and substituent root.")
        if b.GetBondType() != Chem.BondType.SINGLE or b.GetIsAromatic() or b.IsInRing():
            raise ActionError(
                "Source-substituent bond is not a detachable single bond."
            )

        tgt_atom = mol.GetAtomWithIdx(int(tgt))
        if int(tgt_atom.GetTotalNumHs()) <= 0:
            raise ActionError("Target ring atom has no available H.")
        nn = _non_ring_neighbors(tgt_atom, ring_set)
        nn = [x for x in nn if x.GetAtomicNum() > 1]
        if len(nn) != 0:
            raise ActionError("Target ring atom already substituted.")

        rw = Chem.RWMol(Chem.Mol(mol))
        rw.RemoveBond(int(src), int(root))
        rw.AddBond(int(tgt), int(root), Chem.BondType.SINGLE)

        new_mol = rw.GetMol()
        assert_valid_mol(new_mol)

        if int(new_mol.GetNumHeavyAtoms()) < int(self.min_heavy_atoms):
            raise ActionError("Result below min_heavy_atoms.")
        if len(Chem.GetMolFrags(new_mol)) != 1:
            raise ActionError("Result has multiple fragments.")

        after_charge = _total_formal_charge(new_mol)
        if not self.allow_charge_change and before_charge != after_charge:
            raise ActionError("Charge changed and allow_charge_change=False.")

        touched_atoms, touched_bonds = self.touched(mol, action)
        return AppliedAction(
            mol=new_mol,
            action=action,
            touched_atoms=touched_atoms,
            touched_bonds=touched_bonds,
        )
