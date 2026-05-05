from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import BRICS

from ...constraints.base import ConstraintContext
from ...core.errors import ActionError
from ...core.rdkit_utils import assert_valid_mol
from ..base import ActionInstance, ActionOperator, AppliedAction


def _remove_dummy_atoms_and_sanitize(m: Chem.Mol) -> Chem.Mol:
    """Remove BRICS/fragmentation dummy atoms ([#0]) and sanitize."""
    d = Chem.MolFromSmarts("[#0]")
    if d is None:
        raise ActionError("Internal error: could not build dummy SMARTS.")
    out = Chem.DeleteSubstructs(m, d)
    Chem.SanitizeMol(out)
    return out


def _canonical_smiles(m: Chem.Mol) -> str:
    return Chem.MolToSmiles(m, canonical=True)


@dataclass(frozen=True)
class _CutCandidate:
    bond_idx: int
    a: int
    b: int
    keep_atoms: Tuple[int, ...]
    fragment_smiles: str
    contains_anchor: bool
    keep_heavy_atoms: int

    def sort_key(self) -> tuple:
        # Deterministic ordering across molecules/operators.
        return (
            int(self.bond_idx),
            int(min(self.a, self.b)),
            int(max(self.a, self.b)),
            self.fragment_smiles,
        )


class FragmentationOperator(ActionOperator):
    """Enumerate single-bond truncations using RDKit BRICS cut rules.

    This operator performs one BRICS cut (one bond) and returns a single
    truncated fragment suitable for docking.

    Output selection policy (mode):
      - "largest": keep the fragment with the most heavy atoms.
      - "contains_anchor": keep a fragment that contains anchor SMARTS match;
        fallback to "largest" if none.

    Determinism:
      - candidates are sorted by (bond_idx, atom indices, fragment SMILES)
      - returned ActionInstances are sorted by ActionInstance.stable_sort_key().
    """

    name = "FragmentationOperator"

    def __init__(
        self,
        *,
        mode: str = "largest",
        anchor_smarts: Optional[str] = None,
        max_cuts_per_step: int = 1,
        min_heavy_atoms: int = 8,
        max_deleted_atoms: int = 60,
        max_fragments: int = 1,
        method: str = "brics",
        log_library_version: bool = True,
    ) -> None:
        if int(max_cuts_per_step) != 1:
            # Keep scope tight for Tier 2.1.
            raise ValueError(
                "FragmentationOperator currently supports max_cuts_per_step=1 only."
            )

        method_n = str(method).strip().lower()
        if method_n != "brics":
            raise ValueError(
                "FragmentationOperator currently supports method='brics' only."
            )

        mode_n = str(mode).strip().lower()
        if mode_n not in {"largest", "contains_anchor"}:
            raise ValueError("mode must be one of: 'largest', 'contains_anchor'")

        self.mode = mode_n
        self.anchor_smarts = anchor_smarts
        self.max_cuts_per_step = int(max_cuts_per_step)
        self.min_heavy_atoms = int(min_heavy_atoms)
        self.max_deleted_atoms = int(max_deleted_atoms)
        self.max_fragments = int(max_fragments)
        self.method = method_n
        self.log_library_version = bool(log_library_version)

        self._anchor_q = None
        if self.anchor_smarts:
            q = Chem.MolFromSmarts(self.anchor_smarts)
            if q is None:
                raise ValueError(f"Invalid anchor_smarts: {self.anchor_smarts!r}")
            self._anchor_q = q

    def _find_cut_bonds(self, mol: Chem.Mol) -> Iterable[Tuple[int, int, int]]:
        """Yield (bond_idx, a, b) for candidate BRICS cut bonds."""
        for (a, b), _labels in BRICS.FindBRICSBonds(mol):
            bond = mol.GetBondBetweenAtoms(int(a), int(b))
            if bond is None:
                continue
            yield int(bond.GetIdx()), int(a), int(b)

    def _candidates_for_bond(
        self, mol: Chem.Mol, bond_idx: int, a: int, b: int
    ) -> List[_CutCandidate]:
        nA = mol.GetNumAtoms()
        fm = Chem.FragmentOnBonds(mol, [int(bond_idx)], addDummies=True)
        frags_idx = Chem.GetMolFrags(fm, asMols=False, sanitizeFrags=False)
        frags_mol = Chem.GetMolFrags(fm, asMols=True, sanitizeFrags=False)

        out: List[_CutCandidate] = []
        for atom_tuple, frag_m in zip(frags_idx, frags_mol):
            keep_set = sorted(int(i) for i in atom_tuple if int(i) < nA)
            if not keep_set:
                continue

            # Convert to a closed-valence fragment by removing dummy atoms.
            try:
                frag_closed = _remove_dummy_atoms_and_sanitize(frag_m)
            except Exception:
                continue

            heavy = int(frag_closed.GetNumHeavyAtoms())
            if heavy < self.min_heavy_atoms:
                continue

            contains_anchor = False
            if self._anchor_q is not None:
                try:
                    contains_anchor = bool(
                        frag_closed.HasSubstructMatch(self._anchor_q)
                    )
                except Exception:
                    contains_anchor = False

            out.append(
                _CutCandidate(
                    bond_idx=int(bond_idx),
                    a=int(a),
                    b=int(b),
                    keep_atoms=tuple(keep_set),
                    fragment_smiles=_canonical_smiles(frag_closed),
                    contains_anchor=bool(contains_anchor),
                    keep_heavy_atoms=heavy,
                )
            )

        out.sort(key=lambda c: c.sort_key())
        return out

    def _select_candidate(
        self, candidates: List[_CutCandidate]
    ) -> Optional[_CutCandidate]:
        if not candidates:
            return None

        if self.mode == "largest":
            # Max heavy atoms; tie-break by SMILES then by candidate sort_key.
            return sorted(
                candidates,
                key=lambda c: (
                    int(c.keep_heavy_atoms),
                    c.fragment_smiles,
                    c.sort_key(),
                ),
                reverse=True,
            )[0]

        # contains_anchor
        anchored = [c for c in candidates if c.contains_anchor]
        pool = anchored if anchored else candidates
        return sorted(
            pool,
            key=lambda c: (int(c.keep_heavy_atoms), c.fragment_smiles, c.sort_key()),
            reverse=True,
        )[0]

    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:
        actions: List[ActionInstance] = []
        nA = mol.GetNumAtoms()
        if nA < 3:
            return actions

        locked_atoms = ctx.locked_atoms
        locked_bonds = ctx.locked_bonds

        for bond_idx, a, b in self._find_cut_bonds(mol):
            # Respect locked bond (either direction)
            if locked_bonds[(int(a), int(b))] or locked_bonds[(int(b), int(a))]:
                continue

            # Expand to fragment candidates for this cut.
            cands = self._candidates_for_bond(mol, bond_idx=bond_idx, a=a, b=b)
            sel = self._select_candidate(cands)
            if sel is None:
                continue

            keep_atoms = set(int(i) for i in sel.keep_atoms)
            delete_atoms = [int(i) for i in range(nA) if int(i) not in keep_atoms]

            if len(delete_atoms) == 0 or len(delete_atoms) >= nA:
                continue
            if len(delete_atoms) > self.max_deleted_atoms:
                continue

            # Respect locked atoms: never delete locked atoms.
            if any(bool(locked_atoms[int(i)]) for i in delete_atoms):
                continue

            payload = {
                "bond_idx": int(sel.bond_idx),
                "a": int(sel.a),
                "b": int(sel.b),
                "keep_atoms": list(int(i) for i in sel.keep_atoms),
                "delete_atoms": delete_atoms,
                "n_deleted": int(len(delete_atoms)),
                "fragment_smiles": str(sel.fragment_smiles),
                "mode": str(self.mode),
                "method": str(self.method),
                "min_heavy_atoms": int(self.min_heavy_atoms),
                "max_fragments": int(self.max_fragments),
            }

            if self.anchor_smarts is not None:
                payload["anchor_smarts"] = str(self.anchor_smarts)
                payload["contains_anchor"] = bool(sel.contains_anchor)

            if self.log_library_version:
                # Not a user-provided library, but record the fragmentation ruleset.
                payload["library_version"] = "rdkit_brics_v1"

            actions.append(
                ActionInstance(
                    operator=self.name,
                    site=(
                        int(sel.bond_idx),
                        int(min(sel.a, sel.b)),
                        int(max(sel.a, sel.b)),
                    ),
                    template="brics_cut",
                    payload=payload,
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

        for k in ("a", "b"):
            if k in payload:
                j = int(payload[k])
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
            "bond_idx",
            "a",
            "b",
            "delete_atoms",
            "min_heavy_atoms",
            "max_fragments",
        }
        if not required.issubset(payload.keys()):
            raise ActionError("FragmentationOperator missing payload fields.")

        bond_idx = int(payload["bond_idx"])
        a = int(payload["a"])
        b = int(payload["b"])
        delete_atoms = [int(i) for i in payload["delete_atoms"]]
        min_heavy_atoms = int(payload["min_heavy_atoms"])
        max_fragments = int(payload["max_fragments"])

        nA = mol.GetNumAtoms()
        if a < 0 or a >= nA or b < 0 or b >= nA:
            raise ActionError("Invalid atom indices.")
        if bond_idx < 0 or bond_idx >= mol.GetNumBonds():
            raise ActionError("Invalid bond_idx.")

        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None or int(bond.GetIdx()) != bond_idx:
            # Molecule may have changed unexpectedly.
            raise ActionError(
                "Bond between a and b not found (molecule changed unexpectedly)."
            )

        delete_set = set(delete_atoms)
        if len(delete_set) != len(delete_atoms):
            raise ActionError("delete_atoms contains duplicates.")
        if any(i < 0 or i >= nA for i in delete_atoms):
            raise ActionError("delete_atoms contains invalid indices.")
        if len(delete_atoms) == 0 or len(delete_atoms) >= nA:
            raise ActionError("delete_atoms size invalid.")

        # Keep atom: any atom not deleted; use smallest index for fragment selection after deletion.
        keep_atoms = [i for i in range(nA) if i not in delete_set]
        if not keep_atoms:
            raise ActionError("No atoms left after deletion.")
        keep_atom = int(min(keep_atoms))

        # Remove atoms in descending order so indices remain valid.
        rw = Chem.RWMol(mol)
        for idx in sorted(delete_atoms, reverse=True):
            rw.RemoveAtom(int(idx))
        new_mol = rw.GetMol()

        # Map keep_atom to new index after deletions.
        n_deleted_before = sum(1 for x in delete_atoms if int(x) < keep_atom)
        keep_new = int(keep_atom - n_deleted_before)
        if keep_new < 0 or keep_new >= new_mol.GetNumAtoms():
            raise ActionError("Computed keep atom index out of range after deletion.")

        # If multiple fragments remain, keep the one containing keep_new.
        frags_idx = Chem.GetMolFrags(new_mol, asMols=False, sanitizeFrags=False)
        if len(frags_idx) > 1:
            keep_frag_i = None
            for i, frag in enumerate(frags_idx):
                if keep_new in frag:
                    keep_frag_i = i
                    break
            if keep_frag_i is None:
                raise ActionError("Could not identify kept fragment after deletion.")
            frags_mol = Chem.GetMolFrags(new_mol, asMols=True, sanitizeFrags=True)
            new_mol = frags_mol[int(keep_frag_i)]
        else:
            Chem.SanitizeMol(new_mol)

        # Caps
        if new_mol.GetNumHeavyAtoms() < min_heavy_atoms:
            raise ActionError("Result below min_heavy_atoms.")

        n_frags_final = len(
            Chem.GetMolFrags(new_mol, asMols=False, sanitizeFrags=False)
        )
        if n_frags_final > max_fragments:
            raise ActionError(
                f"Fragmentation produced {n_frags_final} fragments (> {max_fragments})."
            )

        assert_valid_mol(new_mol)
        touched_atoms, touched_bonds = self.touched(mol, action)
        return AppliedAction(
            mol=new_mol,
            action=action,
            touched_atoms=touched_atoms,
            touched_bonds=touched_bonds,
        )
