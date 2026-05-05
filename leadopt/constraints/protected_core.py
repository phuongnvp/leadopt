from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from rdkit import Chem

from .base import Constraint, ConstraintContext


@dataclass
class ProtectedCoreConstraint(Constraint):
    """Protect a SMARTS-defined core from being modified by actions.

    This constraint is primarily used for **action gating** via ActionSpace:
      - build(mol) computes the core atom set from `core_smarts` matches and
        marks those atoms as locked in the returned ConstraintContext.
      - is_action_allowed(ctx, touched_atoms, touched_bonds) rejects any action
        that touches a protected core atom.

    Additionally, it provides a **molecule-level** margin via evaluate():
      - margin > 0 if the molecule still contains the core SMARTS
      - margin < 0 if the core is absent (useful for logging/penalty)

    Notes
    -----
    - Deterministic: atom protection uses the union of all substructure matches.
    - Backwards compatible: does not require ActionSpace to use it; when used as
      a standard ConstraintSuite constraint it only evaluates core presence.
    """

    core_smarts: str
    require_match: bool = False
    library_version: str = "protected_core_v1"

    def __post_init__(self) -> None:
        q = Chem.MolFromSmarts(self.core_smarts)
        if q is None:
            raise ValueError(f"Invalid core_smarts: {self.core_smarts!r}")
        self._q = q

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "core_smarts": str(self.core_smarts),
            "require_match": bool(self.require_match),
            "library_version": str(self.library_version),
        }

    def _core_atoms(self, mol: Chem.Mol) -> Set[int]:
        # Union of all matches keeps this deterministic and conservative.
        matches = mol.GetSubstructMatches(self._q)
        core: Set[int] = set()
        for m in matches:
            for a in m:
                core.add(int(a))
        return core

    # -----------------------------
    # ActionSpace gating interface
    # -----------------------------
    def build(self, mol: Chem.Mol) -> ConstraintContext:
        ctx = ConstraintContext()
        core = self._core_atoms(mol)
        ctx.data["protected_core_atoms"] = sorted(core)
        for i in core:
            ctx.locked_atoms[int(i)] = True
        return ctx

    def is_action_allowed(
        self, ctx: ConstraintContext, touched_atoms: set[int], touched_bonds: set[int]
    ) -> bool:
        core_list = ctx.data.get("protected_core_atoms", []) or []
        core = set(int(i) for i in core_list)
        if not core:
            return not bool(self.require_match)
        return core.isdisjoint(set(int(i) for i in touched_atoms))

    # -----------------------------
    # ConstraintSuite interface
    # -----------------------------
    def evaluate(self, mol: Any, context: Optional[Dict[str, Any]] = None) -> float:
        try:
            m = mol
            if not isinstance(m, Chem.Mol):
                return -1.0
            has = bool(m.HasSubstructMatch(self._q))
            if has:
                return 1.0
            return -1.0
        except Exception:
            return -1e6
