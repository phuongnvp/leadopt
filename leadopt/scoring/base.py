from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .types import ScoringResult


class Scorer(ABC):
    """
    Abstract scoring interface.

    Scorers must be:
    - deterministic given the same inputs (as much as feasible)
    - failure-safe: they should never crash the environment; instead return valid=False

    Conventions:
    - objective is always "higher is better"
    - fail_objective is a very low value used when scoring fails
    """

    #: Default fallback objective used for failures (override per scorer if needed).
    fail_objective: float = -1e9

    @property
    def name(self) -> str:
        """Stable scorer identifier for logs/manifests."""
        return self.__class__.__name__

    @property
    def version(self) -> str:
        """
        Optional version string (e.g., model version, docking protocol version).
        Keep stable for reproducibility.
        """
        return "0"

    def scorer_metadata(self) -> Dict[str, Any]:
        """
        Metadata to be logged once per run (config, model path, receptor path, etc.).
        Override in concrete scorers.
        """
        return {"name": self.name, "version": self.version}

    @abstractmethod
    def score(
        self, mol: Any, context: Optional[Dict[str, Any]] = None
    ) -> ScoringResult:
        """
        Score a molecule and return a ScoringResult.

        Parameters
        ----------
        mol:
            Typically an RDKit Mol.
        context:
            Optional dict for passing run-time context (seed, receptor id, budget, etc.).

        Returns
        -------
        ScoringResult
        """
        raise NotImplementedError
