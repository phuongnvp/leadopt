from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from ...constraints.base import ConstraintContext
from ...core.errors import ActionError
from ...core.rdkit_utils import assert_valid_mol
from ..base import ActionInstance, ActionOperator, AppliedAction


def _canonical_smiles(m: Chem.Mol) -> str:
    return Chem.MolToSmiles(m, canonical=True)


def _total_formal_charge(mol: Chem.Mol) -> int:
    return int(sum(int(a.GetFormalCharge()) for a in mol.GetAtoms()))


def _existing_isotopes(mol: Chem.Mol) -> set[int]:
    return {int(a.GetIsotope()) for a in mol.GetAtoms() if int(a.GetIsotope()) != 0}


def _pick_free_isotope(mol: Chem.Mol) -> int:
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
    """Force site-specific reaction firing by adding isotope constraints on mapped reactant atoms."""
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


def _load_yaml(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ActionError(
            f"SMIRKS library YAML must be a mapping, got {type(raw).__name__}."
        )
    return raw


def _require_str(x: Any, where: str) -> str:
    if not isinstance(x, str) or not x:
        raise ActionError(f"{where} must be a non-empty string")
    return x


def _require_list(x: Any, where: str) -> list:
    if not isinstance(x, list):
        raise ActionError(f"{where} must be a list")
    return x


@dataclass(frozen=True)
class SmirksTransform:
    name: str
    smirks: str
    notes: str


@dataclass(frozen=True)
class _CompiledTransform:
    index: int
    t: SmirksTransform
    rxn: rdChemReactions.ChemicalReaction
    reactant_template: Chem.Mol
    reactant_mapnums: Tuple[int, ...]  # aligned with template atom order


class SmirksLibraryOperator(ActionOperator):
    name = "SmirksLibraryOperator"
    DEFAULT_LIBRARY_VERSION = "medchem_smirks_v2"
    DEFAULT_RELATIVE_PATH = "data/smirks/medchem_smirks_v2.yaml"

    def __init__(
        self,
        *,
        library_path: Optional[str | Path] = None,
        min_heavy_atoms: int = 5,
        require_single_fragment: bool = True,
        allow_charge_change: bool = False,
    ) -> None:
        self.min_heavy_atoms = int(min_heavy_atoms)
        self.require_single_fragment = bool(require_single_fragment)
        self.allow_charge_change = bool(allow_charge_change)

        if self.min_heavy_atoms < 1:
            raise ValueError("min_heavy_atoms must be >= 1")

        if library_path is None:
            # Resolve relative to the leadopt package root.
            pkg_root = Path(__file__).resolve().parents[2]
            library_path = pkg_root / self.DEFAULT_RELATIVE_PATH
        else:
            library_path = Path(library_path)

        self.library_path = Path(library_path)
        if not self.library_path.exists():
            raise ActionError(f"SMIRKS library YAML not found: {self.library_path}")

        raw = _load_yaml(self.library_path)
        self.library_version = _require_str(
            raw.get("library_version"), "$.library_version"
        )

        transforms_raw = _require_list(raw.get("transforms"), "$.transforms")
        transforms: List[SmirksTransform] = []
        for i, tr in enumerate(transforms_raw):
            if not isinstance(tr, dict):
                raise ActionError(f"$.transforms[{i}] must be a mapping")
            name = _require_str(tr.get("name"), f"$.transforms[{i}].name")
            smirks = _require_str(tr.get("smirks"), f"$.transforms[{i}].smirks")
            notes = tr.get("notes", "")
            if notes is None:
                notes = ""
            if not isinstance(notes, str):
                raise ActionError(
                    f"$.transforms[{i}].notes must be a string if provided"
                )
            transforms.append(SmirksTransform(name=name, smirks=smirks, notes=notes))

        # Compile and validate SMIRKS in file order (deterministic)
        self._compiled: List[_CompiledTransform] = []
        for idx, t in enumerate(transforms):
            rxn = rdChemReactions.ReactionFromSmarts(str(t.smirks))
            if rxn is None:
                raise ActionError(
                    f"Invalid SMIRKS in {self.library_path} for transform {t.name!r}"
                )
            if rxn.GetNumReactantTemplates() != 1:
                raise ActionError(
                    f"Transform {t.name!r} must have exactly 1 reactant template"
                )

            react = rxn.GetReactantTemplate(0)
            mapnums = tuple(int(a.GetAtomMapNum()) for a in react.GetAtoms())
            if any(mn == 0 for mn in mapnums):
                raise ActionError(
                    f"Transform {t.name!r} must be fully reactant-mapped (no :0 mapnums)"
                )

            self._compiled.append(
                _CompiledTransform(
                    index=idx,
                    t=t,
                    rxn=rxn,
                    reactant_template=react,
                    reactant_mapnums=mapnums,
                )
            )

    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:
        actions: List[ActionInstance] = []

        for ct in self._compiled:
            matches = mol.GetSubstructMatches(ct.reactant_template, uniquify=True)
            matches = sorted(matches)  # deterministic

            for m in matches:
                if any(bool(ctx.locked_atoms[int(i)]) for i in m):
                    continue

                actions.append(
                    ActionInstance(
                        operator=self.name,
                        site=tuple(int(i) for i in m),
                        template=ct.t.name,
                        payload={
                            "library_version": self.library_version,
                            "transform_name": ct.t.name,
                            "transform_index": int(ct.index),
                            "smirks": ct.t.smirks,
                            "notes": ct.t.notes,
                            "match": tuple(int(i) for i in m),
                            "reactant_mapnums": list(ct.reactant_mapnums),
                            "min_heavy_atoms": int(self.min_heavy_atoms),
                            "require_single_fragment": bool(
                                self.require_single_fragment
                            ),
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

        touched_atoms: set[int] = {int(i) for i in action.site}
        touched_bonds: set[int] = set()

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
        reactant_mapnums = payload.get("reactant_mapnums")

        if not isinstance(smirks, str) or not smirks:
            raise ActionError("Missing payload key 'smirks' for SmirksLibraryOperator.")
        if match is None:
            raise ActionError("Missing payload key 'match' for SmirksLibraryOperator.")
        if reactant_mapnums is None:
            raise ActionError(
                "Missing payload key 'reactant_mapnums' for SmirksLibraryOperator."
            )

        match_t = tuple(int(i) for i in match)
        reactant_mapnums = [int(x) for x in reactant_mapnums]

        before_charge = _total_formal_charge(mol)

        iso = _pick_free_isotope(mol)
        rw = Chem.RWMol(Chem.Mol(mol))  # non-mutating
        _label_match_atoms_inplace(rw, match_t, iso)
        reactant_labeled = rw.GetMol()

        smirks_site = _smirks_add_isotope_constraints_to_mapped_atoms(
            smirks, reactant_mapnums, iso
        )
        rxn_site = rdChemReactions.ReactionFromSmarts(smirks_site)
        if rxn_site is None or rxn_site.GetNumReactantTemplates() != 1:
            raise ActionError("Invalid site-specific SMIRKS reaction.")

        try:
            ps = rxn_site.RunReactants((reactant_labeled,))
        except Exception as e:
            raise ActionError(f"SMIRKS apply failed: {e}") from e

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
            raise ActionError("No valid products from SMIRKS.")

        candidates.sort(key=_canonical_smiles)
        new_mol = candidates[0]

        rw_out = Chem.RWMol(Chem.Mol(new_mol))
        _clear_all_isotopes_inplace(rw_out)
        new_mol = rw_out.GetMol()

        assert_valid_mol(new_mol)

        if int(new_mol.GetNumHeavyAtoms()) < int(self.min_heavy_atoms):
            raise ActionError("Result below min_heavy_atoms for SmirksLibraryOperator.")

        if self.require_single_fragment:
            frags = Chem.GetMolFrags(new_mol, asMols=False, sanitizeFrags=False)
            if len(frags) != 1:
                raise ActionError(
                    "Result has multiple fragments and require_single_fragment=True."
                )

        after_charge = _total_formal_charge(new_mol)
        if (not self.allow_charge_change) and before_charge != after_charge:
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
