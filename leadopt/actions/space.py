from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem

from ..constraints.base import ConstraintContext
from ..core.errors import ActionError
from ..core.rules import RuleConfig, check_molecule
from .base import ActionInstance, ActionOperator


@dataclass
class ActionSpace:
    operators: Sequence[ActionOperator]
    constraint: Optional[Any] = (
        None  # gating constraint object (legacy API); may be None
    )
    rule_config: Optional[RuleConfig] = None  # defaults to None = no rule filtering

    def build_context(self, mol: Chem.Mol) -> ConstraintContext:
        if self.constraint is None:
            return ConstraintContext()
        return self.constraint.build(mol)

    def _payload_to_stable_str(self, payload: Optional[dict]) -> str:
        payload = payload or {}
        try:
            return json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        except Exception:
            # Fallback if payload contains non-JSON-serializable objects
            return repr(payload)

    def _site_to_stable_str(self, site) -> str:
        # site can be int, tuple, None, etc.
        try:
            return json.dumps(
                site, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        except Exception:
            return repr(site)

    def _action_sort_key(self, a: ActionInstance) -> Tuple[str, str, str, str]:
        return (
            str(a.operator),
            self._site_to_stable_str(a.site),
            str(a.template),
            self._payload_to_stable_str(a.payload),
        )

    def enumerate(self, mol: Chem.Mol) -> List[ActionInstance]:
        ctx = self.build_context(mol)
        all_actions: List[ActionInstance] = []
        for op in self.operators:
            all_actions.extend(list(op.enumerate_actions(mol, ctx)))

        # Ensure deterministic action ordering across runs/platforms
        all_actions.sort(key=self._action_sort_key)
        return all_actions

    def filter_allowed(
        self, mol: Chem.Mol, actions: Sequence[ActionInstance]
    ) -> List[ActionInstance]:
        """
        Backwards-compatible API: returns only the allowed actions.
        Internally uses filter_allowed_with_applied() and discards applied mol cache.
        """
        allowed, _applied = self.filter_allowed_with_applied(mol, actions)
        return allowed

    def filter_allowed_with_applied(
        self, mol: Chem.Mol, actions: Sequence[ActionInstance]
    ) -> Tuple[List[ActionInstance], List[Optional[Chem.Mol]]]:
        """
        Returns:
          allowed: list of allowed ActionInstance (subset of `actions`)
          applied_mols: list aligned with `actions`; entry i is the resulting Chem.Mol if allowed else None

        This method applies each candidate action at most once and caches the resulting molecule,
        enabling the environment to reuse it in step() (avoids double-apply and mask/step mismatch).
        """
        ctx = self.build_context(mol)
        allowed: List[ActionInstance] = []
        applied_mols: List[Optional[Chem.Mol]] = [None] * len(actions)

        op_map: Dict[str, ActionOperator] = {op.name: op for op in self.operators}

        for i, a in enumerate(actions):
            op = op_map.get(a.operator)
            if op is None:
                continue

            # 1) constraint gate (optional)
            touched_atoms, touched_bonds = op.touched(mol, a)
            if self.constraint is not None:
                try:
                    if not self.constraint.is_action_allowed(
                        ctx, touched_atoms, touched_bonds
                    ):
                        continue
                except Exception:
                    # If gating constraint errors, treat as disallowed (conservative)
                    continue

            # Always allow terminate as an escape hatch (no need to apply)
            if a.operator == "Terminate" or bool(
                (a.payload or {}).get("terminate", False)
            ):
                allowed.append(a)
                applied_mols[i] = Chem.Mol(mol)  # next mol is identical
                continue

            # Optional cheap feasibility gate (only if operator implements it)
            # This should NOT modify mol and should NOT sanitize.
            is_feasible = getattr(op, "is_feasible", None)
            if callable(is_feasible):
                try:
                    if not is_feasible(mol, a, ctx):  # type: ignore[misc]
                        continue
                except Exception:
                    # If feasibility check itself errors, treat as not feasible
                    continue

            # 2) chemistry gate (apply ONCE here)
            try:
                applied = op.apply(mol, a)
            except ActionError:
                continue
            except Exception:
                continue

            # Defensive: ensure we got a molecule back
            next_mol = getattr(applied, "mol", None)
            if next_mol is None:
                continue

            # 3) optional rule gate
            if self.rule_config is not None:
                try:
                    ok, _reasons = check_molecule(next_mol, self.rule_config)
                except Exception:
                    continue
                if not ok:
                    continue

            allowed.append(a)
            applied_mols[i] = Chem.Mol(
                next_mol
            )  # copy to decouple from operator internals

        return allowed, applied_mols
