from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from ...constraints.base import ConstraintContext
from ...core.errors import ActionError
from ...core.rdkit_utils import assert_valid_mol
from ..base import ActionInstance, ActionOperator, AppliedAction


def _canonical_smiles(m: Chem.Mol) -> str:
    return Chem.MolToSmiles(m, canonical=True)


def _existing_isotopes(mol: Chem.Mol) -> set[int]:
    return {int(a.GetIsotope()) for a in mol.GetAtoms() if int(a.GetIsotope()) != 0}


def _pick_free_isotope(mol: Chem.Mol) -> int:
    """
    Deterministically pick an isotope label not present in `mol`.

    Uses a high range to avoid collisions with any realistic isotope usage.
    """
    used = _existing_isotopes(mol)
    for iso in range(200, 1000):
        if iso not in used:
            return iso
    raise ActionError("No free isotope label available for site-disambiguation.")


def _label_match_atoms_inplace(
    rw: Chem.RWMol, match: tuple[int, ...], iso: int
) -> None:
    for idx in match:
        rw.GetAtomWithIdx(int(idx)).SetIsotope(int(iso))


def _clear_all_isotopes_inplace(rw: Chem.RWMol) -> None:
    for a in rw.GetAtoms():
        if int(a.GetIsotope()) != 0:
            a.SetIsotope(0)


def _smirks_add_isotope_constraints_to_mapped_atoms(
    smirks: str, reactant_mapnums: list[int], iso: int
) -> str:
    """
    Add isotope constraints to the REACTANT side for the specified atom-map numbers.

    Example (iso=200, mapnums=[1,2]):
      [C:1][C:2]>>[C:1][O:2]
      -> [200C:1][200C:2]>>[C:1][O:2]

    If an atom already has an isotope, it is replaced with `iso` for determinism.
    """
    parts = smirks.split(">>")
    if len(parts) != 2:
        raise ActionError("SMIRKS must contain exactly one '>>' separator.")
    lhs, rhs = parts[0], parts[1]

    for mnum in reactant_mapnums:
        pat = re.compile(rf"\[([^\]]*?):{int(mnum)}\]")

        def _repl(matchobj: re.Match[str]) -> str:
            inner = matchobj.group(1)
            inner2 = re.sub(r"^\d+", str(int(iso)), inner)
            if inner2 == inner:
                inner2 = f"{int(iso)}{inner}"
            return f"[{inner2}:{int(mnum)}]"

        lhs, n = pat.subn(_repl, lhs)
        if n == 0:
            raise ActionError(
                f"SMIRKS reactant side missing mapped atom :{int(mnum)} required for site selection."
            )

    return f"{lhs}>>{rhs}"


@dataclass(frozen=True)
class _RxnEntry:
    rid: str
    smirks: str
    rxn: rdChemReactions.ChemicalReaction
    reactant_mapnums: Optional[
        Tuple[int, ...]
    ]  # aligned with reactant template atom order


class ReactionSMARTSOperator(ActionOperator):
    """
    Apply RDKit chemical reactions defined by SMIRKS.

    Determinism:
      - enumerate_actions sorts matches and actions by stable_sort_key
      - apply selects deterministic product (canonical SMILES min)

    Site consistency:
      - If the SMIRKS reactant template uses atom-mapping on ALL atoms (mapnums != 0),
        enumerate_actions stores reactant_mapnums aligned with the match tuple order.
      - apply() then enforces that the reaction fires ONLY at that match by:
          * labeling match atoms with a free isotope
          * rewriting SMIRKS reactant-side mapped atoms to require that isotope
          * running the site-specific SMIRKS on the labeled reactant
      - If mapping is missing, apply() falls back to legacy behavior (global product selection).

    Contracts:
      - apply is non-mutating (operates on copies)
      - touched is consistent with action.site
    """

    name = "ReactionSMARTSOperator"

    def __init__(self, reactions: Sequence[Tuple[str, str]]) -> None:
        """
        reactions: list of (rid, smirks)
        """
        self._rxns: List[_RxnEntry] = []

        for rid, smirks in reactions:
            rxn = rdChemReactions.ReactionFromSmarts(str(smirks))
            if rxn is None:
                raise ActionError(f"Invalid SMIRKS for reaction {rid!r}.")
            if rxn.GetNumReactantTemplates() != 1:
                raise ActionError(
                    f"Reaction {rid!r} must have exactly 1 reactant template."
                )

            react = rxn.GetReactantTemplate(0)
            # Mapnums aligned with reactant template atom order (same order as SubstructMatch tuples).
            mapnums = tuple(int(a.GetAtomMapNum()) for a in react.GetAtoms())
            if any(mn == 0 for mn in mapnums):
                reactant_mapnums: Optional[Tuple[int, ...]] = None
            else:
                reactant_mapnums = mapnums

            self._rxns.append(
                _RxnEntry(
                    rid=str(rid),
                    smirks=str(smirks),
                    rxn=rxn,
                    reactant_mapnums=reactant_mapnums,
                )
            )

    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:
        actions: List[ActionInstance] = []

        for entry in self._rxns:
            react = entry.rxn.GetReactantTemplate(0)
            matches = mol.GetSubstructMatches(react, uniquify=True)
            matches = sorted(matches)  # deterministic

            for m in matches:
                if any(bool(ctx.locked_atoms[int(i)]) for i in m):
                    continue

                payload = {"smirks": entry.smirks, "match": tuple(int(i) for i in m)}
                if entry.reactant_mapnums is not None:
                    payload["reactant_mapnums"] = list(entry.reactant_mapnums)

                actions.append(
                    ActionInstance(
                        operator=self.name,
                        site=tuple(int(i) for i in m),
                        template=entry.rid,
                        payload=payload,
                    )
                )

        actions.sort(key=lambda a: a.stable_sort_key())
        return actions

    def touched(
        self, mol: Chem.Mol, action: ActionInstance
    ) -> tuple[set[int], set[int]]:
        self._ensure_operator_match(action)

        touched_atoms: set[int] = {int(i) for i in action.site}
        touched_bonds: set[int] = set()

        # Conservative but simple: include bonds between any two atoms in the match if present.
        mol_n = mol.GetNumAtoms()
        for i in touched_atoms:
            if i < 0 or i >= mol_n:
                continue
            ai = mol.GetAtomWithIdx(int(i))
            for b in ai.GetBonds():
                a = int(b.GetBeginAtomIdx())
                c = int(b.GetEndAtomIdx())
                if a in touched_atoms and c in touched_atoms:
                    touched_bonds.add(int(b.GetIdx()))

        return touched_atoms, touched_bonds

    def apply(self, mol: Chem.Mol, action: ActionInstance) -> AppliedAction:
        self._ensure_operator_match(action)
        payload = action.payload or {}

        smirks = payload.get("smirks")
        match = payload.get("match")
        if not smirks or match is None:
            raise ActionError(
                "Missing smirks/match payload for ReactionSMARTSOperator."
            )

        match_t = tuple(int(i) for i in match)

        # Prefer site-consistent apply if mapping info is present.
        reactant_mapnums = payload.get("reactant_mapnums")
        if reactant_mapnums is not None:
            reactant_mapnums = [int(x) for x in reactant_mapnums]

            # Label this match in a copied reactant
            iso = _pick_free_isotope(mol)
            rw = Chem.RWMol(Chem.Mol(mol))  # non-mutating copy
            _label_match_atoms_inplace(rw, match_t, iso)
            reactant_labeled = rw.GetMol()

            # Rewrite SMIRKS to require isotope at the mapped atoms (reactant side)
            smirks_site = _smirks_add_isotope_constraints_to_mapped_atoms(
                str(smirks), reactant_mapnums, iso
            )
            rxn_site = rdChemReactions.ReactionFromSmarts(smirks_site)
            if rxn_site is None or rxn_site.GetNumReactantTemplates() != 1:
                raise ActionError("Invalid site-specific SMIRKS reaction.")

            try:
                ps = rxn_site.RunReactants((reactant_labeled,))
            except Exception as e:
                raise ActionError(f"Reaction application failed: {e}") from e

            candidates: List[Chem.Mol] = []
            for prod_tuple in ps:
                for p in prod_tuple:
                    if p is None:
                        continue
                    try:
                        assert_valid_mol(p)
                    except Exception:
                        continue
                    candidates.append(p)

            if not candidates:
                raise ActionError(
                    "No valid products from site-specific ReactionSMARTS apply()."
                )

            candidates.sort(key=_canonical_smiles)
            new_mol = candidates[0]

            # Clear any isotopes that survive into the product (conservative cleanup)
            rw_out = Chem.RWMol(Chem.Mol(new_mol))
            _clear_all_isotopes_inplace(rw_out)
            new_mol = rw_out.GetMol()

        else:
            # Backwards-compatible legacy behavior (no mapping => cannot enforce a specific match site)
            rxn = rdChemReactions.ReactionFromSmarts(str(smirks))
            if rxn is None or rxn.GetNumReactantTemplates() != 1:
                raise ActionError("Invalid or unsupported SMIRKS reaction.")

            try:
                ps = rxn.RunReactants((Chem.Mol(mol),))
            except Exception as e:
                raise ActionError(f"Reaction application failed: {e}") from e

            candidates = []
            for prod_tuple in ps:
                for p in prod_tuple:
                    if p is None:
                        continue
                    try:
                        assert_valid_mol(p)
                    except Exception:
                        continue
                    candidates.append(p)

            if not candidates:
                raise ActionError("No valid products from ReactionSMARTS.")

            candidates.sort(key=_canonical_smiles)
            new_mol = candidates[0]

        assert_valid_mol(new_mol)
        touched_atoms, touched_bonds = self.touched(mol, action)
        return AppliedAction(
            mol=new_mol,
            action=action,
            touched_atoms=touched_atoms,
            touched_bonds=touched_bonds,
        )
