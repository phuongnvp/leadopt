from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from rdkit import Chem

from ...constraints.base import ConstraintContext
from ...core.errors import ActionError
from ...core.rdkit_utils import assert_valid_mol
from ..base import ActionInstance, ActionOperator, AppliedAction

# -------------------------------------------------------------------------
# R-group libraries (single attachment via one dummy atom [*:1])
# Keep lists ordered to preserve determinism.
#
# Academic reproducibility rule:
# - Never mutate an existing versioned library in-place.
# - Introduce new versions (medchem_v2, ...) for expansions.
# -------------------------------------------------------------------------

# Original demo library from early leadopt versions.
MEDCHEM_RGROUP_LIBRARY_V0: List[Tuple[str, str]] = [
    ("Me", "[*:1]C"),
    ("Et", "[*:1]CC"),
    ("iPr", "[*:1]C(C)C"),
    ("tBu", "[*:1]C(C)(C)C"),
    ("F", "[*:1]F"),
    ("Cl", "[*:1]Cl"),
    ("Br", "[*:1]Br"),
    ("I", "[*:1]I"),
    ("OH", "[*:1]O"),
    ("OMe", "[*:1]OC"),
    ("NH2", "[*:1]N"),
    ("CN", "[*:1]C#N"),
    ("CF3", "[*:1]C(F)(F)F"),
    ("CONH2", "[*:1]C(=O)N"),
    ("SO2Me", "[*:1]S(=O)(=O)C"),
]

# Backwards-compatibility alias (historical name).
RGROUP_LIBRARY: List[Tuple[str, str]] = MEDCHEM_RGROUP_LIBRARY_V0

# Expanded medchem R-group library (Tier 1.1).
MEDCHEM_RGROUP_LIBRARY_V1: List[Tuple[str, str]] = [
    # Simple alkyl / halogens
    ("Me", "[*:1]C"),
    ("Et", "[*:1]CC"),
    ("nPr", "[*:1]CCC"),
    ("iPr", "[*:1]C(C)C"),
    ("nBu", "[*:1]CCCC"),
    ("tBu", "[*:1]C(C)(C)C"),
    ("iBu", "[*:1]CC(C)C"),
    ("secBu", "[*:1]C(C)CC"),
    ("cPr", "[*:1]C1CC1"),
    ("cBu", "[*:1]C1CCC1"),
    ("cPent", "[*:1]C1CCCC1"),
    ("cHex", "[*:1]C1CCCCC1"),
    ("F", "[*:1]F"),
    ("Cl", "[*:1]Cl"),
    ("Br", "[*:1]Br"),
    ("I", "[*:1]I"),
    # Small polar / HBD-HBA
    ("OH", "[*:1]O"),
    ("OMe", "[*:1]OC"),
    ("OEt", "[*:1]OCC"),
    ("O-iPr", "[*:1]OC(C)C"),
    ("O-tBu", "[*:1]OC(C)(C)C"),
    ("SH", "[*:1]S"),
    ("SMe", "[*:1]SC"),
    ("SEt", "[*:1]SCC"),
    ("CN", "[*:1]C#N"),
    ("CF3", "[*:1]C(F)(F)F"),
    ("CHF2", "[*:1]C(F)F"),
    ("CH2F", "[*:1]CF"),
    ("CH2CF3", "[*:1]CC(F)(F)F"),
    ("CH2CHF2", "[*:1]CC(F)F"),
    ("OCF3", "[*:1]OC(F)(F)F"),
    ("SCF3", "[*:1]SC(F)(F)F"),
    # Carbonyl-containing (common medchem vectors)
    ("CHO", "[*:1]C=O"),
    ("COMe", "[*:1]C(=O)C"),
    ("COEt", "[*:1]C(=O)CC"),
    ("CO2Me", "[*:1]C(=O)OC"),
    ("CO2Et", "[*:1]C(=O)OCC"),
    ("CONH2", "[*:1]C(=O)N"),
    ("CONHMe", "[*:1]C(=O)NC"),
    ("CONMe2", "[*:1]C(=O)N(C)C"),
    ("CO2H", "[*:1]C(=O)O"),
    ("CONHEt", "[*:1]C(=O)NCC"),
    ("CONHCH2OH", "[*:1]C(=O)NCO"),
    ("CONHCH2CH2OH", "[*:1]C(=O)NCCO"),
    ("SO2Me", "[*:1]S(=O)(=O)C"),
    ("SO2Et", "[*:1]S(=O)(=O)CC"),
    ("SO2NH2", "[*:1]S(=O)(=O)N"),
    ("SO2NHMe", "[*:1]S(=O)(=O)NC"),
    ("SO2NMe2", "[*:1]S(=O)(=O)N(C)C"),
    ("SO2NH-iPr", "[*:1]S(=O)(=O)NC(C)C"),
    ("SO2NHPh", "[*:1]S(=O)(=O)Nc1ccccc1"),
    # Small heterocycles / solubilizers
    ("Morpholine", "[*:1]N1CCOCC1"),
    ("Piperidine", "[*:1]N1CCCCC1"),
    ("Piperazine", "[*:1]N1CCNCC1"),
    ("Azetidine", "[*:1]N1CCC1"),
    ("Pyrrolidine", "[*:1]N1CCCC1"),
    ("Oxetane", "[*:1]C1COC1"),
    ("THF", "[*:1]C1CCOC1"),
    ("THP", "[*:1]C1CCCCO1"),
    # Aromatic/heteroaromatic fragments (single attachment)
    ("Ph", "[*:1]c1ccccc1"),
    ("Bn", "[*:1]Cc1ccccc1"),
    ("Pyridyl-2", "[*:1]c1ccccn1"),
    ("Pyridyl-3", "[*:1]c1cccnc1"),
    ("Pyridyl-4", "[*:1]c1ccncc1"),
    ("Thiazolyl", "[*:1]c1nccs1"),
    ("Oxazolyl", "[*:1]c1ncco1"),
    ("Imidazolyl", "[*:1]c1ncc[nH]1"),
    ("1,2,4-Triazolyl", "[*:1]n1cnnc1"),
    ("Thiophen-3-yl", "[*:1]c1ccsc1"),
    ("Fur-3-yl", "[*:1]c1ccoc1"),
    ("Thiophen-2-yl", "[*:1]c1sccc1"),
    ("Fur-2-yl", "[*:1]c1occc1"),
    ("Pyrimidyl", "[*:1]c1nccnc1"),
    ("Pyrazinyl", "[*:1]c1cnccn1"),
    ("Pyridazinyl", "[*:1]n1ncccc1"),
    # Neutral Amines (avoid fixed charges)
    ("CH2OH", "[*:1]CO"),
    ("CH2OMe", "[*:1]COC"),
    ("CH2NH2", "[*:1]CN"),
    ("CH2NMe2", "[*:1]CN(C)C"),
    ("CH2CONH2", "[*:1]CC(=O)N"),
    ("NMe2", "[*:1]N(C)C"),
    ("NEt2", "[*:1]N(CC)CC"),
    ("Dimethylaminoethyl", "[*:1]CCN(C)C"),
    ("NH2", "[*:1]N"),
    ("NHMe", "[*:1]NC"),
]

MEDCHEM_LIBRARY_VERSION_V0 = "medchem_fragments_v0"
MEDCHEM_LIBRARY_VERSION_V1 = "medchem_fragments_v1"


def _count_heavy_atoms(m: Chem.Mol) -> int:
    return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1)


def _find_single_dummy(m: Chem.Mol) -> int:
    dummies = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1:
        raise ActionError("Template must contain exactly one dummy atom [*].")
    return int(dummies[0])


def _set_orig_idx_props(m: Chem.Mol, prop: str = "_orig_idx") -> None:
    # Preserve original atom indices through fragmentation
    for a in m.GetAtoms():
        a.SetIntProp(prop, a.GetIdx())


def _get_atom_by_orig_idx(
    m: Chem.Mol, orig_idx: int, prop: str = "_orig_idx"
) -> Chem.Atom:
    for a in m.GetAtoms():
        if a.HasProp(prop) and int(a.GetIntProp(prop)) == int(orig_idx):
            return a
    raise ActionError(f"Could not locate atom with {prop}={orig_idx} in fragment.")


class RGroupSwap(ActionOperator):

    name = "RGroupSwap"

    def __init__(
        self,
        *,
        max_sidechain_heavy_atoms: int = 12,
        library: str = "medchem_v1",
        include_library_version_in_payload: bool = True,
        fragment_subset: Optional[Sequence[str]] = None,
        max_actions_per_bond: Optional[int] = None,
        deduplicate_products: bool = False,
    ) -> None:

        self.max_sidechain_heavy_atoms = int(max_sidechain_heavy_atoms)
        self.include_library_version_in_payload = bool(
            include_library_version_in_payload
        )

        self.fragment_subset = (
            list(fragment_subset) if fragment_subset is not None else None
        )
        self.max_actions_per_bond = max_actions_per_bond
        self.deduplicate_products = bool(deduplicate_products)

        lib = str(library)
        if lib == "medchem_v1":
            self.rgroup_library = list(MEDCHEM_RGROUP_LIBRARY_V1)
            self.library_version = MEDCHEM_LIBRARY_VERSION_V1
        elif lib == "medchem_v0":
            self.rgroup_library = list(MEDCHEM_RGROUP_LIBRARY_V0)
            self.library_version = MEDCHEM_LIBRARY_VERSION_V0
        else:
            raise ValueError(f"Unknown RGroupSwap library: {lib!r}")

        # Deterministic subset filtering
        if self.fragment_subset is not None:
            allowed = set(self.fragment_subset)
            self.rgroup_library = [
                tpl for tpl in self.rgroup_library if tpl[0] in allowed
            ]

    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:

        actions: List[ActionInstance] = []
        seen_products = set() if self.deduplicate_products else None

        locked_bonds = getattr(ctx, "locked_bonds", None)
        locked_atoms = getattr(ctx, "locked_atoms", None)

        def _is_locked_atom(idx: int) -> bool:
            if locked_atoms is None:
                return False
            try:
                return bool(locked_atoms[idx])  # dict/list/defaultdict
            except Exception:
                # Some ConstraintContext variants store a Mol or other non-indexable object here.
                # In that case, we conservatively treat as "unlocked" rather than crash.
                return False

        for bond in mol.GetBonds():
            bidx = bond.GetIdx()

            if locked_bonds is not None and bool(locked_bonds[bidx]):
                continue

            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            if bond.GetIsAromatic():
                continue
            if bond.IsInRing():
                continue

            a = bond.GetBeginAtom()
            b = bond.GetEndAtom()
            aidx = a.GetIdx()
            bidx_atom = b.GetIdx()

            base = Chem.Mol(mol)
            _set_orig_idx_props(base)

            frag = Chem.FragmentOnBonds(base, [bond.GetIdx()], addDummies=True)
            frags_idx = Chem.GetMolFrags(frag, asMols=False, sanitizeFrags=False)
            frags_mols = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)

            if len(frags_idx) != 2:
                continue

            sizes = [_count_heavy_atoms(fm) for fm in frags_mols]
            small_k = 0 if sizes[0] <= sizes[1] else 1
            big_k = 1 - small_k

            if sizes[small_k] > self.max_sidechain_heavy_atoms:
                continue

            big_orig_idxs = set(frags_idx[big_k])
            attach = (
                aidx
                if aidx in big_orig_idxs
                else (bidx_atom if bidx_atom in big_orig_idxs else None)
            )
            if attach is None:
                continue

            if _is_locked_atom(aidx) and _is_locked_atom(bidx_atom):
                continue

            per_bond_count = 0

            for tpl_id, tpl_smiles in self.rgroup_library:

                if self.max_actions_per_bond is not None:
                    if per_bond_count >= self.max_actions_per_bond:
                        break

                action = ActionInstance(
                    operator=self.name,
                    site=(int(attach), int(bond.GetIdx())),
                    template=tpl_id,
                    payload={
                        "attach_atom": int(attach),
                        "bond_idx": int(bond.GetIdx()),
                        "tpl_smiles": tpl_smiles,
                        "max_sidechain_heavy_atoms": int(
                            self.max_sidechain_heavy_atoms
                        ),
                        **(
                            {"library_version": self.library_version}
                            if (
                                self.include_library_version_in_payload
                                and self.library_version is not None
                            )
                            else {}
                        ),
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
                per_bond_count += 1

        actions.sort(key=lambda a: a.stable_sort_key())
        return actions

    def touched(
        self, mol: Chem.Mol, action: ActionInstance
    ) -> tuple[set[int], set[int]]:
        self._ensure_operator_match(action)
        payload = action.payload or {}
        attach = int(payload.get("attach_atom", action.site[0]))
        bond_idx = int(payload.get("bond_idx", action.site[1]))
        return {attach}, {bond_idx}

    def apply(self, mol: Chem.Mol, action: ActionInstance) -> AppliedAction:
        self._ensure_operator_match(action)
        payload = action.payload or {}

        if (
            "attach_atom" not in payload
            or "bond_idx" not in payload
            or "tpl_smiles" not in payload
        ):
            raise ActionError("Missing required payload keys for RGroupSwap.")

        attach_atom_orig = int(payload["attach_atom"])
        bond_idx = int(payload["bond_idx"])
        tpl_smiles = str(payload["tpl_smiles"])

        if attach_atom_orig < 0 or attach_atom_orig >= mol.GetNumAtoms():
            raise ActionError("Invalid attach_atom index.")
        if bond_idx < 0 or bond_idx >= mol.GetNumBonds():
            raise ActionError("Invalid bond_idx.")

        # Clone and annotate original indices for robust mapping through fragmentation
        base = Chem.Mol(mol)
        _set_orig_idx_props(base)

        # Fragment the target bond; keep the fragment containing attach_atom_orig
        frag = Chem.FragmentOnBonds(base, [bond_idx], addDummies=True)
        # frags_idx = Chem.GetMolFrags(frag, asMols=False, sanitizeFrags=False)
        frags_mols = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)

        if len(frags_mols) != 2:
            raise ActionError("Bond fragmentation did not yield two fragments.")

        # Pick the fragment that contains attach atom (by original index property)
        scaffold = None
        for fm in frags_mols:
            try:
                _ = _get_atom_by_orig_idx(fm, attach_atom_orig)
                scaffold = fm
                break
            except ActionError:
                continue

        if scaffold is None:
            raise ActionError(
                "Could not find scaffold fragment containing attachment atom."
            )

        # Identify the scaffold dummy connected to attach atom
        attach_atom = _get_atom_by_orig_idx(scaffold, attach_atom_orig)
        attach_idx = int(attach_atom.GetIdx())

        dummy_scaffold_idx = None
        for nb in attach_atom.GetNeighbors():
            if nb.GetAtomicNum() == 0:
                dummy_scaffold_idx = int(nb.GetIdx())
                break
        if dummy_scaffold_idx is None:
            raise ActionError(
                "No scaffold dummy found adjacent to attachment atom after fragmentation."
            )

        # Build substituent template
        sub = Chem.MolFromSmiles(tpl_smiles)
        if sub is None:
            raise ActionError(f"Failed to parse template SMILES: {tpl_smiles!r}")
        dummy_sub_idx = _find_single_dummy(sub)

        # Root atom of substituent is the neighbor of dummy
        dummy_sub_atom = sub.GetAtomWithIdx(dummy_sub_idx)
        if dummy_sub_atom.GetDegree() != 1:
            raise ActionError("Template dummy atom must have exactly one neighbor.")
        sub_root_idx = int(dummy_sub_atom.GetNeighbors()[0].GetIdx())

        # Combine scaffold + substituent
        combo = Chem.CombineMols(scaffold, sub)
        rw = Chem.RWMol(combo)
        off = scaffold.GetNumAtoms()

        attach_idx_c = attach_idx
        dummy_scaffold_idx_c = dummy_scaffold_idx
        dummy_sub_idx_c = off + dummy_sub_idx
        sub_root_idx_c = off + sub_root_idx

        # Add bond attach -> substituent root
        rw.AddBond(attach_idx_c, sub_root_idx_c, Chem.BondType.SINGLE)

        # Remove dummy atoms (remove larger index first)
        to_remove = sorted([dummy_sub_idx_c, dummy_scaffold_idx_c], reverse=True)
        for ridx in to_remove:
            rw.RemoveAtom(int(ridx))

        new_mol = rw.GetMol()
        assert_valid_mol(new_mol)

        touched_atoms, touched_bonds = self.touched(mol, action)
        return AppliedAction(
            mol=new_mol,
            action=action,
            touched_atoms=touched_atoms,
            touched_bonds=touched_bonds,
        )
