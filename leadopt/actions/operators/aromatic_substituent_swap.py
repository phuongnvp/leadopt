from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from rdkit import Chem

from ...constraints.base import ConstraintContext
from ...core.errors import ActionError
from ...core.rdkit_utils import assert_valid_mol
from ..base import ActionInstance, ActionOperator, AppliedAction
from .r_group_swap import (
    MEDCHEM_LIBRARY_VERSION_V1,
    MEDCHEM_RGROUP_LIBRARY_V1,
)


def _total_formal_charge(mol: Chem.Mol) -> int:
    return int(sum(int(a.GetFormalCharge()) for a in mol.GetAtoms()))


def _is_aromatic_atom(atom: Chem.Atom) -> bool:
    return atom.GetIsAromatic() and atom.IsInRing()


def _get_non_ring_neighbors(atom: Chem.Atom) -> List[Chem.Atom]:
    return [n for n in atom.GetNeighbors() if not n.IsInRing()]


def _is_polar_fragment_smiles(smiles: str) -> bool:
    """
    Very simple polarity heuristic:
      - polar if contains N/O/S/P (not counting halogens).
    Deterministic and conservative; opt-in only.
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False
    for a in m.GetAtoms():
        z = int(a.GetAtomicNum())
        if z in (7, 8, 15, 16):  # N, O, P, S
            return True
    return False


def _set_orig_idx_props(m: Chem.Mol, prop: str = "_orig_idx") -> None:
    for a in m.GetAtoms():
        a.SetIntProp(prop, a.GetIdx())


def _fragment_contains_orig_idx(
    m: Chem.Mol, orig_idx: int, prop: str = "_orig_idx"
) -> bool:
    for a in m.GetAtoms():
        if a.HasProp(prop) and int(a.GetIntProp(prop)) == int(orig_idx):
            return True
    return False


def _canonicalize_dummy_fragment_smiles(smiles: str) -> str:
    """
    Canonicalize a fragment SMILES (possibly with [*:1]) for identity comparison:
      - parse
      - clear atom map numbers
      - canonical smiles
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return ""
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m, canonical=True)


def _current_substituent_fragment_smiles(
    mol: Chem.Mol, ring_idx: int, sub_idx: int
) -> str:
    """
    Return canonical SMILES (with dummy) of the current substituent attached to ring_idx-sub_idx.
    Uses FragmentOnBonds(addDummies=True). Clears atom maps to make comparable to library fragments.
    """
    bond = mol.GetBondBetweenAtoms(int(ring_idx), int(sub_idx))
    if bond is None:
        return ""

    base = Chem.Mol(mol)
    _set_orig_idx_props(base)

    fragged = Chem.FragmentOnBonds(base, [bond.GetIdx()], addDummies=True)
    frags = Chem.GetMolFrags(fragged, asMols=True, sanitizeFrags=False)
    if len(frags) != 2:
        return ""

    # Identify which frag contains the ring atom (by orig idx prop), return the other
    ring_frag_idx = None
    for k, fm in enumerate(frags):
        if _fragment_contains_orig_idx(fm, ring_idx):
            ring_frag_idx = k
            break
    if ring_frag_idx is None:
        return ""

    sub_frag = frags[1 - ring_frag_idx]

    # Clear any atom maps (shouldn't have, but keep consistent)
    for a in sub_frag.GetAtoms():
        a.SetAtomMapNum(0)

    return Chem.MolToSmiles(sub_frag, canonical=True)


@dataclass(frozen=True)
class _SwapSite:
    ring_atom_idx: int
    substituent_root_idx: int
    substituent_is_polar: bool


class AromaticSubstituentSwap(ActionOperator):
    """
    Tier 3.3 — Replace a single non-ring substituent on an aromatic ring atom
    with a fragment from a versioned R-group library.

    Tier 4.1.3 additions (opt-in, default off):
      - fragment_subset
      - max_actions_per_site
      - skip_if_same_fragment
      - forbid_polar_to_polar
      - deduplicate_products (by product canonical SMILES during enumeration)

    Constraints:
      - ring core unchanged
      - only mono-substituted positions
      - single fragment result
    """

    name = "AromaticSubstituentSwap"

    def __init__(
        self,
        *,
        min_heavy_atoms: int = 5,
        allow_charge_change: bool = False,
        fragment_subset: Optional[Sequence[str]] = None,
        max_actions_per_site: Optional[int] = None,
        skip_if_same_fragment: bool = False,
        forbid_polar_to_polar: bool = False,
        deduplicate_products: bool = False,
    ) -> None:
        self.fragments = list(MEDCHEM_RGROUP_LIBRARY_V1)
        self.library_version = MEDCHEM_LIBRARY_VERSION_V1

        self.min_heavy_atoms = int(min_heavy_atoms)
        self.allow_charge_change = bool(allow_charge_change)

        self.fragment_subset = (
            list(fragment_subset) if fragment_subset is not None else None
        )
        self.max_actions_per_site = max_actions_per_site
        self.skip_if_same_fragment = bool(skip_if_same_fragment)
        self.forbid_polar_to_polar = bool(forbid_polar_to_polar)
        self.deduplicate_products = bool(deduplicate_products)

        # Deterministic subset filtering: preserve library order
        if self.fragment_subset is not None:
            allowed = set(self.fragment_subset)
            self.fragments = [tpl for tpl in self.fragments if tpl[0] in allowed]

    def _enumerate_sites(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> List[_SwapSite]:
        sites: List[_SwapSite] = []

        locked_atoms = getattr(ctx, "locked_atoms", None)

        def _is_locked_atom(idx: int) -> bool:
            if locked_atoms is None:
                return False
            try:
                return bool(locked_atoms[idx])
            except Exception:
                # Some ConstraintContext variants store a Mol or other non-indexable object here.
                # Treat as unlocked rather than crash.
                return False

        for atom in mol.GetAtoms():
            idx = int(atom.GetIdx())
            if _is_locked_atom(idx):
                continue
            if not _is_aromatic_atom(atom):
                continue

            non_ring_neighbors = _get_non_ring_neighbors(atom)
            if len(non_ring_neighbors) != 1:
                continue  # mono-substituent only

            root = non_ring_neighbors[0]
            root_idx = int(root.GetIdx())

            # Polarity heuristic on the current substituent (approx by atom types in subtree root)
            # Conservative: consider polar if root atom or its immediate neighbors include N/O/S/P.
            substituent_is_polar = False
            stack = [root_idx]
            seen = set()
            rw = Chem.Mol(mol)
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                a = rw.GetAtomWithIdx(cur)
                z = int(a.GetAtomicNum())
                if z in (7, 8, 15, 16):
                    substituent_is_polar = True
                    break
                for nb in a.GetNeighbors():
                    ni = int(nb.GetIdx())
                    if ni == idx:
                        continue
                    if ni not in seen:
                        stack.append(ni)

            sites.append(
                _SwapSite(
                    ring_atom_idx=idx,
                    substituent_root_idx=root_idx,
                    substituent_is_polar=substituent_is_polar,
                )
            )
        return sites

    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:
        actions: List[ActionInstance] = []
        sites = self._enumerate_sites(mol, ctx)

        seen_products = set() if self.deduplicate_products else None

        for site in sites:
            per_site_count = 0

            # Compute current substituent identity once per site if needed
            current_sub_smiles = ""
            if self.skip_if_same_fragment:
                current_sub_smiles = _current_substituent_fragment_smiles(
                    mol, site.ring_atom_idx, site.substituent_root_idx
                )

            for frag_idx, (frag_name, frag_smiles) in enumerate(self.fragments):
                if (
                    self.max_actions_per_site is not None
                    and per_site_count >= self.max_actions_per_site
                ):
                    break

                if self.skip_if_same_fragment and current_sub_smiles:
                    # Compare canonicalized forms with atom maps cleared
                    lib_can = _canonicalize_dummy_fragment_smiles(frag_smiles)
                    cur_can = _canonicalize_dummy_fragment_smiles(current_sub_smiles)
                    if lib_can and cur_can and lib_can == cur_can:
                        continue

                if self.forbid_polar_to_polar:
                    frag_is_polar = _is_polar_fragment_smiles(frag_smiles)
                    if site.substituent_is_polar and frag_is_polar:
                        continue

                action = ActionInstance(
                    operator=self.name,
                    site=(site.ring_atom_idx,),
                    template=frag_name,
                    payload={
                        "library_version": self.library_version,
                        "fragment_name": frag_name,
                        "fragment_index": int(frag_idx),
                        "fragment_smiles": frag_smiles,
                        "ring_atom_idx": int(site.ring_atom_idx),
                        "substituent_root_idx": int(site.substituent_root_idx),
                        "min_heavy_atoms": int(self.min_heavy_atoms),
                        "allow_charge_change": bool(self.allow_charge_change),
                    },
                )

                if self.deduplicate_products:
                    try:
                        applied = self.apply(mol, action)
                        smiles = Chem.MolToSmiles(applied.mol, canonical=True)
                    except Exception:
                        continue
                    if smiles in seen_products:
                        continue
                    seen_products.add(smiles)

                actions.append(action)
                per_site_count += 1

        actions.sort(key=lambda a: a.stable_sort_key())
        return actions

    def touched(
        self, mol: Chem.Mol, action: ActionInstance
    ) -> tuple[set[int], set[int]]:
        ring_idx = int(action.payload["ring_atom_idx"])
        sub_idx = int(action.payload["substituent_root_idx"])
        return {ring_idx, sub_idx}, set()

    def apply(self, mol: Chem.Mol, action: ActionInstance) -> AppliedAction:
        self._ensure_operator_match(action)
        payload = action.payload or {}

        ring_idx = int(payload["ring_atom_idx"])
        sub_idx = int(payload["substituent_root_idx"])
        frag_smiles = payload["fragment_smiles"]

        before_charge = _total_formal_charge(mol)

        # --- Step 1: remove substituent subtree ---
        rw = Chem.RWMol(Chem.Mol(mol))

        to_delete = {sub_idx}
        frontier = [sub_idx]

        while frontier:
            current = frontier.pop()
            atom = rw.GetAtomWithIdx(current)
            for n in atom.GetNeighbors():
                ni = int(n.GetIdx())
                if ni == ring_idx:
                    continue
                if ni not in to_delete:
                    to_delete.add(ni)
                    frontier.append(ni)

        # RDKit renumbers atom indices on deletion. Compute the post-deletion ring index.
        # Any deleted atom with index < ring_idx shifts ring_idx down by 1.
        n_lower = sum(1 for d in to_delete if int(d) < int(ring_idx))
        ring_idx_new = int(ring_idx) - int(n_lower)

        for idx in sorted(to_delete, reverse=True):
            rw.RemoveAtom(int(idx))

        core_mol = rw.GetMol()

        # --- Step 2: prepare fragment ---
        frag = Chem.MolFromSmiles(frag_smiles)
        if frag is None:
            raise ActionError("Invalid fragment SMILES in AromaticSubstituentSwap.")

        rw_frag = Chem.RWMol(frag)

        dummy_idx = None
        for a in rw_frag.GetAtoms():
            if a.GetAtomicNum() == 0:
                dummy_idx = int(a.GetIdx())
                break

        if dummy_idx is None:
            raise ActionError("Fragment missing attachment dummy atom.")

        dummy_atom = rw_frag.GetAtomWithIdx(dummy_idx)
        neighbors = list(dummy_atom.GetNeighbors())
        if len(neighbors) != 1:
            raise ActionError("Attachment dummy must have exactly one neighbor.")

        frag_attach_neighbor_idx = int(neighbors[0].GetIdx())

        # --- Step 3: combine core + fragment ---
        combined = Chem.CombineMols(core_mol, rw_frag.GetMol())
        rw_combined = Chem.RWMol(combined)

        core_n = core_mol.GetNumAtoms()
        frag_attach_idx = core_n + frag_attach_neighbor_idx
        frag_dummy_idx = core_n + dummy_idx

        rw_combined.AddBond(ring_idx_new, frag_attach_idx, Chem.rdchem.BondType.SINGLE)
        rw_combined.RemoveAtom(frag_dummy_idx)

        new_mol = rw_combined.GetMol()
        assert_valid_mol(new_mol)

        if int(new_mol.GetNumHeavyAtoms()) < self.min_heavy_atoms:
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
