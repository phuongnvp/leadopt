from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from rdkit import Chem

from ..constraints.base import ConstraintContext
from ..core.errors import ActionError
from ..core.rdkit_utils import assert_valid_mol, clone_mol


@dataclass(frozen=True)
class ActionInstance:
    """
    Concrete action to be applied to a molecule.
    'payload' is operator-specific and must be serializable (JSON-friendly) for logging.
    """

    operator: str
    site: Tuple[int, ...]  # e.g., (atom_idx,) or (bond_idx,) etc.
    template: Optional[str] = None  # e.g., R-group id, swap id, mutation label
    payload: Dict[str, Any] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", self.payload or {})

    def stable_payload_json(self) -> str:
        """
        Deterministic JSON representation of payload for sorting and logging.
        Falls back to repr() for non-JSON-serializable objects (should be avoided).
        """
        import json

        return json.dumps(
            self.payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=repr,
        )

    def stable_sort_key(self) -> tuple:
        """
        Stable, total ordering key for deterministic enumerate_actions().
        Includes payload JSON to break ties where site/template are equal.
        """
        return (
            self.operator,
            tuple(int(i) for i in self.site),
            self.template or "",
            self.stable_payload_json(),
        )


@dataclass
class AppliedAction:
    """
    Result of applying an action.
    """

    mol: Chem.Mol
    action: ActionInstance
    touched_atoms: set[int]
    touched_bonds: set[int]


class ActionOperator(ABC):
    """
    Base class for all medicinal chemistry operators.
    Operators must:
      - enumerate valid ActionInstances given a molecule + constraint context
      - apply an ActionInstance deterministically using RDKit transformations
      - declare which atoms/bonds they intend to touch (for constraint enforcement)
    """

    name: str

    @abstractmethod
    def enumerate_actions(
        self, mol: Chem.Mol, ctx: ConstraintContext
    ) -> Sequence[ActionInstance]:
        raise NotImplementedError

    @abstractmethod
    def touched(
        self, mol: Chem.Mol, action: ActionInstance
    ) -> tuple[set[int], set[int]]:
        """
        Return (touched_atoms, touched_bonds) for constraint checking.
        Should be conservative (include all possibly affected indices).
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, mol: Chem.Mol, action: ActionInstance) -> AppliedAction:
        """
        Apply the action to the molecule. Must raise ActionError on failure.
        Must return a sanitized, valid molecule.
        """
        raise NotImplementedError

    def _ensure_operator_match(self, action: ActionInstance) -> None:
        if action.operator != self.name:
            raise ActionError(
                f"Action operator mismatch: expected {self.name!r} got {action.operator!r}"
            )

    def _prepare_working_copy(self, mol: Chem.Mol) -> Chem.Mol:
        m = clone_mol(mol)
        assert_valid_mol(m)
        return m

    # ----------------------------
    # Operator contract (Phase 7.1)
    # ----------------------------
    # Determinism requirement:
    #   - enumerate_actions() must return a deterministic ordering for a given (mol, ctx)
    #   - apply() must be deterministic (no hidden randomness)
    #   - ActionInstance payloads SHOULD be JSON-serializable (for logging + signature stability)
    #
    # Signature-guard requirement:
    #   - repr(op) must be stable across processes for the same operator configuration.
    #     (Used by scripts/train_ppo.py and scripts/1_compound.py.)

    def __repr__(self) -> str:  # pragma: no cover
        cfg = self._signature_config()
        try:
            cfg_s = json.dumps(
                cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        except Exception:
            cfg_s = repr(cfg)
        return f"{self.__class__.__name__}({cfg_s})"

    def _signature_config(self) -> Dict[str, Any]:
        """Return a JSON-friendly config dict used for stable repr/signatures.

        Default implementation introspects instance attributes and normalizes common
        container types. Subclasses may override for tighter control.
        """

        def _norm(x: Any) -> Any:
            if x is None or isinstance(x, (bool, int, float, str)):
                return x
            if isinstance(x, tuple):
                return [_norm(v) for v in x]
            if isinstance(x, list):
                return [_norm(v) for v in x]
            if isinstance(x, dict):
                return {str(k): _norm(v) for k, v in x.items()}
            if isinstance(x, set):
                return sorted([_norm(v) for v in x], key=lambda v: repr(v))
            # RDKit mols/atoms/bonds and other non-JSON objects should not appear here.
            # Fall back to repr (stable enough for basic builtins).
            return repr(x)

        return {
            str(k): _norm(v)
            for k, v in self.__dict__.items()
            if not str(k).startswith("_")
        }

    def is_feasible(self, mol, action, ctx) -> bool:
        return True
