from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ConstraintContext:
    """
    Context used by the *action-gating* constraint system (ActionSpace/operators).

    Many operators expect:
      - ctx.locked_atoms[i] -> bool
      - ctx.locked_bonds[(i, j)] -> bool

    So we implement these as defaultdict(bool) for safe indexing.
    """

    locked_atoms: Any = field(default_factory=lambda: defaultdict(bool))
    locked_bonds: Any = field(default_factory=lambda: defaultdict(bool))
    data: Dict[str, Any] = field(default_factory=dict)


class Constraint(ABC):
    """
    Constraint interface returning a scalar *margin*:

        margin > 0  : satisfied
        margin == 0 : on the boundary
        margin < 0  : violated (magnitude = severity)

    Constraints should be:
    - cheap
    - deterministic (as much as possible)
    - failure-safe (never crash training; return a strongly negative margin if needed)
    """

    @property
    def name(self) -> str:
        """Stable identifier for logging / YAML presets."""
        return self.__class__.__name__

    def metadata(self) -> Dict[str, Any]:
        """
        Metadata describing the constraint configuration (for run manifests).
        Override in subclasses.
        """
        return {"name": self.name}

    @abstractmethod
    def evaluate(self, mol: Any, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate constraint margin on molecule.

        Parameters
        ----------
        mol:
            Typically an RDKit Mol.
        context:
            Optional dict: can contain lead mol/fp, run info, etc.

        Returns
        -------
        margin: float
        """
        raise NotImplementedError
