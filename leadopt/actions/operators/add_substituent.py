from __future__ import annotations

from typing import List, Optional, Sequence

from rdkit import Chem

from ...constraints.base import ConstraintContext
from ...core.errors import ActionError
from ...core.rdkit_utils import assert_valid_mol
from ..base import ActionInstance, ActionOperator, AppliedAction


class AddSubstituent(ActionOperator):
    """
    Attach a small substituent fragment at an atom with an available H.
    Uses fragments with a single dummy atom [*:1] marking the attachment point.

    Conservative: single bonds only; small library; avoids locked-atom editing by enumeration.
    """

    name = "AddSubstituent"

    # ---------------------------------------------------------------------
    # Fragment libraries (single attachment via one dummy atom [*:1])
    # Keep lists ordered to preserve determinism.
    # ---------------------------------------------------------------------

    MEDCHEM_TEMPLATES_V0 = [
        ("Me", "[*:1]C"),
        ("Et", "[*:1]CC"),
        ("F", "[*:1]F"),
        ("Cl", "[*:1]Cl"),
        ("OH", "[*:1]O"),
        ("NH2", "[*:1]N"),
    ]

    MEDCHEM_TEMPLATES_V1 = [
        ("Me", "[*:1]C"),
        ("Et", "[*:1]CC"),
        ("nPr", "[*:1]CCC"),
        ("iPr", "[*:1]C(C)C"),
        ("nBu", "[*:1]CCCC"),
        ("tBu", "[*:1]C(C)(C)C"),
        ("cPr", "[*:1]C1CC1"),
        ("cBu", "[*:1]C1CCC1"),
        ("cPent", "[*:1]C1CCCC1"),
        ("cHex", "[*:1]C1CCCCC1"),
        ("F", "[*:1]F"),
        ("Cl", "[*:1]Cl"),
        ("Br", "[*:1]Br"),
        ("I", "[*:1]I"),
        ("OH", "[*:1]O"),
        ("OMe", "[*:1]OC"),
        ("OEt", "[*:1]OCC"),
        ("O-iPr", "[*:1]OC(C)C"),
        ("SH", "[*:1]S"),
        ("SMe", "[*:1]SC"),
        ("SEt", "[*:1]SCC"),
        ("CN", "[*:1]C#N"),
        ("CF3", "[*:1]C(F)(F)F"),
        ("CHF2", "[*:1]C(F)F"),
        ("OCF3", "[*:1]OC(F)(F)F"),
        ("CH2F", "[*:1]CF"),
        ("NH2", "[*:1]N"),
        ("NHMe", "[*:1]NC"),
        ("NMe2", "[*:1]N(C)C"),
        ("NEt2", "[*:1]N(CC)CC"),
        ("NMeEt", "[*:1]N(C)CC"),
        ("CHO", "[*:1]C=O"),
        ("COMe", "[*:1]C(=O)C"),
        ("CO2Me", "[*:1]C(=O)OC"),
        ("CO2Et", "[*:1]C(=O)OCC"),
        ("CONH2", "[*:1]C(=O)N"),
        ("CONHMe", "[*:1]C(=O)NC"),
        ("CONMe2", "[*:1]C(=O)N(C)C"),
        ("SO2Me", "[*:1]S(=O)(=O)C"),
        ("SO2NH2", "[*:1]S(=O)(=O)N"),
        ("SO2NHMe", "[*:1]S(=O)(=O)NC"),
        ("SO2NMe2", "[*:1]S(=O)(=O)N(C)C"),
        ("CO2H", "[*:1]C(=O)O"),
        ("CO2tBu", "[*:1]C(=O)OC(C)(C)C"),
        ("CONHEt", "[*:1]C(=O)NCC"),
        ("NHAc", "[*:1]NC(=O)C"),
        ("Carbamate-OMe", "[*:1]NC(=O)OC"),
        ("Urea-NH2", "[*:1]NC(=O)N"),
        ("Morpholine", "[*:1]N1CCOCC1"),
        ("Piperidine", "[*:1]N1CCCCC1"),
        ("Piperazine", "[*:1]N1CCNCC1"),
        ("Azetidine", "[*:1]N1CCC1"),
        ("Pyrrolidine", "[*:1]N1CCCC1"),
        ("Oxetane", "[*:1]C1COC1"),
        ("THF", "[*:1]C1CCOC1"),
        ("Ph", "[*:1]c1ccccc1"),
        ("Bn", "[*:1]Cc1ccccc1"),
        ("Pyridyl-2", "[*:1]c1ccccn1"),
        ("Pyridyl-3", "[*:1]c1cccnc1"),
        ("Pyridyl-4", "[*:1]c1ccncc1"),
        ("Imidazole", "[*:1]c1ncc[nH]1"),
        ("1,2,4-Triazole", "[*:1]n1cnnc1"),
        ("Thiophenyl-3", "[*:1]c1ccsc1"),
        ("Thiophenyl-2", "[*:1]c1sccc1"),
        ("Furanyl-3", "[*:1]c1ccoc1"),
        ("Furanyl-2", "[*:1]c1occc1"),
        ("Pyrimidyl", "[*:1]c1nccnc1"),
        ("Pyrazinyl", "[*:1]c1cnccn1"),
    ]

    DEFAULT_TEMPLATES = MEDCHEM_TEMPLATES_V1
    MEDCHEM_LIBRARY_VERSION_V0 = "medchem_fragments_v0"
    MEDCHEM_LIBRARY_VERSION_V1 = "medchem_fragments_v1"

    def __init__(
        self,
        templates: Sequence[tuple[str, str]] | None = None,
        *,
        library: str = "medchem_v1",
        include_library_version_in_payload: bool = True,
        fragment_subset: Optional[Sequence[str]] = None,
        max_actions_per_atom: Optional[int] = None,
        deduplicate_products: bool = False,
    ) -> None:

        self.include_library_version_in_payload = bool(
            include_library_version_in_payload
        )
        self.fragment_subset = (
            list(fragment_subset) if fragment_subset is not None else None
        )
        self.max_actions_per_atom = max_actions_per_atom
        self.deduplicate_products = bool(deduplicate_products)

        if templates is not None:
            self.templates = list(templates)
            self.library_version = None
            return

        lib = str(library)
        if lib == "medchem_v1":
            self.templates = list(self.MEDCHEM_TEMPLATES_V1)
            self.library_version = self.MEDCHEM_LIBRARY_VERSION_V1
        elif lib == "medchem_v0":
            self.templates = list(self.MEDCHEM_TEMPLATES_V0)
            self.library_version = self.MEDCHEM_LIBRARY_VERSION_V0
        else:
            raise ValueError(f"Unknown AddSubstituent library: {lib!r}")

        # Apply fragment subset filtering (deterministic, preserve order)
        if self.fragment_subset is not None:
            allowed = set(self.fragment_subset)
            self.templates = [tpl for tpl in self.templates if tpl[0] in allowed]

    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:
        actions: List[ActionInstance] = []
        seen_products = set() if self.deduplicate_products else None

        for atom in mol.GetAtoms():
            i = atom.GetIdx()

            if atom.GetAtomicNum() == 1:
                continue
            if atom.GetTotalNumHs() <= 0:
                continue
            if atom.GetFormalCharge() != 0:
                continue

            per_atom_count = 0

            for name, frag in self.templates:

                if self.max_actions_per_atom is not None:
                    if per_atom_count >= self.max_actions_per_atom:
                        break

                action = ActionInstance(
                    operator=self.name,
                    site=(i,),
                    template=name,
                    payload={
                        "frag_smiles": frag,
                        "attach_atom_idx": i,
                        "attach_z": atom.GetAtomicNum(),
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
                per_atom_count += 1

        actions.sort(key=lambda a: a.stable_sort_key())
        return actions

    def touched(
        self, mol: Chem.Mol, action: ActionInstance
    ) -> tuple[set[int], set[int]]:
        attach = None
        if action.payload is not None:
            attach = action.payload.get("attach_atom_idx", None)
        if attach is None and action.site:
            attach = int(action.site[0])

        touched_atoms: set[int] = set()
        touched_bonds: set[int] = set()

        if attach is not None and 0 <= int(attach) < mol.GetNumAtoms():
            touched_atoms.add(int(attach))

        return touched_atoms, touched_bonds

    def apply(self, mol: Chem.Mol, action: ActionInstance) -> AppliedAction:
        self._ensure_operator_match(action)
        site = int(action.site[0])
        payload = action.payload or {}
        frag_smiles = payload.get("frag_smiles")
        if not frag_smiles:
            raise ActionError("Missing fragment SMILES.")

        if site < 0 or site >= mol.GetNumAtoms():
            raise ActionError("Invalid attachment site.")

        frag = Chem.MolFromSmiles(frag_smiles, sanitize=True)
        if frag is None:
            raise ActionError("Invalid fragment SMILES.")
        assert_valid_mol(frag)

        dummy_idxs = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
        if len(dummy_idxs) != 1:
            raise ActionError("Fragment must contain exactly one dummy atom [*].")
        d_idx = dummy_idxs[0]

        d_atom = frag.GetAtomWithIdx(d_idx)
        if d_atom.GetDegree() != 1:
            raise ActionError("Dummy atom must have degree 1.")

        frag_nbr = d_atom.GetNeighbors()[0].GetIdx()

        combo = Chem.CombineMols(mol, frag)
        rw = Chem.RWMol(combo)

        offset = mol.GetNumAtoms()
        d_idx_c = offset + d_idx
        nbr_idx_c = offset + frag_nbr

        rw.AddBond(site, nbr_idx_c, Chem.BondType.SINGLE)
        rw.RemoveAtom(d_idx_c)

        new_mol = rw.GetMol()
        assert_valid_mol(new_mol)

        t_atoms, t_bonds = self.touched(mol, action)
        return AppliedAction(
            mol=new_mol, action=action, touched_atoms=t_atoms, touched_bonds=t_bonds
        )
