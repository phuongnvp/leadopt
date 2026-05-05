from __future__ import annotations

from typing import List, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from ...constraints.base import ConstraintContext
from ...core.errors import ActionError
from ...core.rdkit_utils import assert_valid_mol
from ..base import ActionInstance, ActionOperator, AppliedAction

# Ordered list for determinism.
# Keep these conservative and common; expand later.
BIOISOSTERE_SMIRKS: List[Tuple[str, str]] = [
    # Ester -> Amide (very common)
    ("ester_to_amide", "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[N:3][C:4]"),
    # Amide -> Ester (reverse; useful for exploration)
    ("amide_to_ester", "[C:1](=[O:2])[N:3][C:4]>>[C:1](=[O:2])[O:3][C:4]"),
    # Thioether -> Ether
    ("thioether_to_ether", "[C:1][S:2][C:3]>>[C:1][O:2][C:3]"),
    # Ether -> Thioether
    ("ether_to_thioether", "[C:1][O:2][C:3]>>[C:1][S:2][C:3]"),
]


def _canonical_smiles(m: Chem.Mol) -> str:
    return Chem.MolToSmiles(m, canonical=True)


class BioisostereSwap(ActionOperator):
    """
    Apply a curated set of single-reactant SMIRKS transformations (bioisosteres / interconversions).

    Conservative v1:
      - single reactant -> single product reactions only
      - enumerate matches deterministically
      - pick deterministic product if multiple outcomes occur

    Core safety:
      - we exclude matches that include locked atoms
    """

    name = "BioisostereSwap"

    def __init__(self, library: Sequence[Tuple[str, str]] | None = None) -> None:
        self.library = (
            list(library) if library is not None else list(BIOISOSTERE_SMIRKS)
        )
        self._rxns = [
            (rid, smirks, rdChemReactions.ReactionFromSmarts(smirks))
            for rid, smirks in self.library
        ]
        for rid, smirks, rxn in self._rxns:
            if rxn is None:
                raise ValueError(f"Failed to parse SMIRKS for {rid}: {smirks!r}")
            if rxn.GetNumReactantTemplates() != 1:
                raise ValueError(
                    f"{rid} must be single-reactant SMIRKS for this operator."
                )

    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:
        actions: List[ActionInstance] = []

        for rid, smirks, rxn in self._rxns:
            react = rxn.GetReactantTemplate(0)
            matches = mol.GetSubstructMatches(react, uniquify=True)
            # deterministic: RDKit provides stable ordering; we still sort by match tuple
            matches = sorted(matches)

            for m in matches:
                # core safety: skip if any locked atom participates
                if any(bool(ctx.locked_atoms[int(i)]) for i in m):
                    continue

                actions.append(
                    ActionInstance(
                        operator=self.name,
                        site=tuple(int(i) for i in m),
                        template=rid,
                        payload={"smirks": smirks, "match": tuple(int(i) for i in m)},
                    )
                )

        actions.sort(key=lambda a: a.stable_sort_key())
        return actions

    def touched(
        self, mol: Chem.Mol, action: ActionInstance
    ) -> tuple[set[int], set[int]]:
        self._ensure_operator_match(action)
        return set(int(i) for i in action.site), set()

    def apply(self, mol: Chem.Mol, action: ActionInstance) -> AppliedAction:
        self._ensure_operator_match(action)
        payload = action.payload or {}
        smirks = payload.get("smirks")
        match = payload.get("match")

        if not smirks or match is None:
            raise ActionError("Missing smirks/match payload.")
        match = tuple(int(i) for i in match)

        rxn = rdChemReactions.ReactionFromSmarts(str(smirks))
        if rxn is None or rxn.GetNumReactantTemplates() != 1:
            raise ActionError("Invalid or unsupported SMIRKS reaction.")

        # Run reaction; RDKit may return multiple product sets
        try:
            ps = rxn.RunReactants((Chem.Mol(mol),))
        except Exception as e:
            raise ActionError(f"Reaction application failed: {e}")

        # Flatten products, sanitize, and choose deterministically
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
            raise ActionError("No valid products from bioisostere swap.")

        # Deterministic pick: smallest canonical SMILES
        candidates.sort(key=_canonical_smiles)
        new_mol = candidates[0]
        assert_valid_mol(new_mol)

        touched_atoms, touched_bonds = self.touched(mol, action)
        return AppliedAction(
            mol=new_mol, action=action, touched_atoms=touched_atoms, touched_bonds=set()
        )
