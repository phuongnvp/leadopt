from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from rdkit import Chem

from ...constraints.base import ConstraintContext
from ...core.errors import ActionError
from ...core.rdkit_utils import assert_valid_mol
from ..base import ActionInstance, ActionOperator, AppliedAction


def _total_formal_charge(mol: Chem.Mol) -> int:
    return int(sum(int(a.GetFormalCharge()) for a in mol.GetAtoms()))


def _atomic_num_to_symbol(z: int) -> str:
    # RDKit provides periodic table; fall back to numeric string if anything weird happens.
    try:
        pt = Chem.GetPeriodicTable()
        sym = pt.GetElementSymbol(int(z))
        return str(sym)
    except Exception:
        return str(z)


@dataclass(frozen=True)
class _SwapSpec:
    atom_idx: int
    from_z: int
    to_z: int


class LinkerAtomSwap(ActionOperator):
    """
    Tier 3.1: Conservative linker internal atom swaps.

    Targets:
      - degree-2 atoms
      - non-ring, non-aromatic
      - both adjacent bonds are SINGLE
      - (by default) neutral atoms only (formal charge == 0)

    Replacements:
      - atomic numbers in allowed_atomic_nums, default (6, 7, 8, 16) = C, N, O, S
      - swapping among them, excluding no-op

    Safety gates (apply-time):
      - RDKit sanitize + connected (assert_valid_mol)
      - minimum heavy atoms (min_heavy_atoms)
      - (optional) forbid net formal charge change (allow_charge_change=False default)

    Determinism:
      - enumerate_actions returns ActionInstances sorted by ActionInstance.stable_sort_key().
    """

    name = "LinkerAtomSwap"

    def __init__(
        self,
        *,
        allowed_atomic_nums: Tuple[int, ...] = (6, 7, 8, 16),
        min_heavy_atoms: int = 5,
        allow_charge_change: bool = False,
        require_neutral_atom: bool = True,
    ) -> None:
        self.allowed_atomic_nums = tuple(int(z) for z in allowed_atomic_nums)
        self.min_heavy_atoms = int(min_heavy_atoms)
        self.allow_charge_change = bool(allow_charge_change)
        self.require_neutral_atom = bool(require_neutral_atom)

        if len(self.allowed_atomic_nums) < 2:
            raise ValueError("allowed_atomic_nums must contain at least 2 elements.")
        if self.min_heavy_atoms < 1:
            raise ValueError("min_heavy_atoms must be >= 1.")

    def _is_candidate_atom(self, mol: Chem.Mol, atom: Chem.Atom) -> bool:
        if atom.GetAtomicNum() == 1:
            return False
        if atom.GetIsAromatic():
            return False
        if atom.IsInRing():
            return False
        if int(atom.GetDegree()) != 2:
            return False
        if self.require_neutral_atom and int(atom.GetFormalCharge()) != 0:
            return False

        # Only consider linker-like atoms: both bonds must be SINGLE
        bonds = list(atom.GetBonds())
        if len(bonds) != 2:
            return False
        if any(b.GetBondType() != Chem.rdchem.BondType.SINGLE for b in bonds):
            return False

        # Restrict to atoms in the allowed set (prevents e.g. swapping halogens)
        if int(atom.GetAtomicNum()) not in self.allowed_atomic_nums:
            return False

        return True

    def _enumerate_swap_specs(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Iterable[_SwapSpec]:
        for atom in mol.GetAtoms():
            i = int(atom.GetIdx())
            if ctx.locked_atoms[i]:
                continue
            if not self._is_candidate_atom(mol, atom):
                continue

            from_z = int(atom.GetAtomicNum())
            for to_z in self.allowed_atomic_nums:
                to_z = int(to_z)
                if to_z == from_z:
                    continue
                yield _SwapSpec(atom_idx=i, from_z=from_z, to_z=to_z)

    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:
        actions: List[ActionInstance] = []

        for spec in self._enumerate_swap_specs(mol, ctx):
            from_sym = _atomic_num_to_symbol(spec.from_z)
            to_sym = _atomic_num_to_symbol(spec.to_z)
            actions.append(
                ActionInstance(
                    operator=self.name,
                    site=(int(spec.atom_idx),),
                    template=f"{from_sym}->{to_sym}",
                    payload={
                        "atom_idx": int(spec.atom_idx),
                        "from": int(spec.from_z),
                        "to": int(spec.to_z),
                        "allowed_atomic_nums": list(self.allowed_atomic_nums),
                        "min_heavy_atoms": int(self.min_heavy_atoms),
                        "allow_charge_change": bool(self.allow_charge_change),
                        "require_neutral_atom": bool(self.require_neutral_atom),
                    },
                )
            )

        actions.sort(key=lambda a: a.stable_sort_key())
        return actions

    def touched(
        self, mol: Chem.Mol, action: ActionInstance
    ) -> tuple[set[int], set[int]]:
        self._ensure_operator_match(action)

        atom_idx = int(action.site[0])
        if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
            # Conservative: return empty touched on invalid index; apply() will raise.
            return set(), set()

        atom = mol.GetAtomWithIdx(atom_idx)

        touched_atoms = {atom_idx}
        touched_bonds: set[int] = set()

        # Conservative: include neighbors + incident bonds
        for b in atom.GetBonds():
            touched_bonds.add(int(b.GetIdx()))
            touched_atoms.add(int(b.GetBeginAtomIdx()))
            touched_atoms.add(int(b.GetEndAtomIdx()))

        return touched_atoms, touched_bonds

    def apply(self, mol: Chem.Mol, action: ActionInstance) -> AppliedAction:
        self._ensure_operator_match(action)

        payload = action.payload or {}
        if "to" not in payload:
            raise ActionError("Missing payload key 'to' for LinkerAtomSwap.")
        atom_idx = int(action.site[0])
        to_z = int(payload["to"])

        if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
            raise ActionError("Invalid atom index for LinkerAtomSwap.")

        # Re-check candidate criteria at apply-time (important for safety)
        atom0 = mol.GetAtomWithIdx(atom_idx)
        if not self._is_candidate_atom(mol, atom0):
            raise ActionError(
                "Target atom is not a valid linker candidate for LinkerAtomSwap."
            )
        if to_z not in self.allowed_atomic_nums:
            raise ActionError("Target atomic number not allowed for LinkerAtomSwap.")

        before_charge = _total_formal_charge(mol)

        rw = Chem.RWMol(mol)
        atom = rw.GetAtomWithIdx(atom_idx)
        atom.SetAtomicNum(int(to_z))

        new_mol = rw.GetMol()

        # Validate + sanitize + connectivity
        try:
            assert_valid_mol(new_mol)
        except Exception as e:
            raise ActionError(f"LinkerAtomSwap produced invalid molecule: {e}") from e

        # Apply-time safety gates
        if int(new_mol.GetNumHeavyAtoms()) < int(self.min_heavy_atoms):
            raise ActionError("Result below min_heavy_atoms for LinkerAtomSwap.")

        after_charge = _total_formal_charge(new_mol)
        if (not self.allow_charge_change) and (before_charge != after_charge):
            raise ActionError(
                f"Net formal charge changed ({before_charge} -> {after_charge}) and allow_charge_change=False."
            )

        touched_atoms, touched_bonds = self.touched(mol, action)
        return AppliedAction(
            mol=new_mol,
            action=action,
            touched_atoms=touched_atoms,
            touched_bonds=touched_bonds,
        )
