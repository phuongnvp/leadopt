from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from rdkit import Chem

from .base import Scorer
from .types import ScoringResult


class LegacyFunctionScorer(Scorer):
    """
    Adapter for the old scoring interface: score_fn(smiles) -> float.

    - Converts mol -> canonical SMILES
    - Calls user-provided score_fn(smiles)
    - Wraps result in ScoringResult
    - Failure-safe: catches exceptions and returns valid=False with fail_objective
    """

    def __init__(
        self,
        score_fn: Callable[[str], float],
        *,
        fail_objective: Optional[float] = None,
        name: str = "LegacyFunctionScorer",
        version: str = "0",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._score_fn = score_fn
        self._name = name
        self._version = version
        self._extra_metadata = dict(extra_metadata or {})
        if fail_objective is not None:
            # allow override of base class default
            self.fail_objective = float(fail_objective)

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def scorer_metadata(self) -> Dict[str, Any]:
        md = {"name": self.name, "version": self.version, "type": "legacy_function"}
        md.update(self._extra_metadata)
        return md

    def score(
        self, mol: Any, context: Optional[Dict[str, Any]] = None
    ) -> ScoringResult:
        try:
            if mol is None:
                return ScoringResult(
                    objective=float(self.fail_objective),
                    components={"objective": float(self.fail_objective)},
                    valid=False,
                    fail_reason="mol_is_none",
                    metadata={**self.scorer_metadata(), "exception_type": None},
                )

            # Convert to canonical SMILES
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            if not smiles:
                return ScoringResult(
                    objective=float(self.fail_objective),
                    components={"objective": float(self.fail_objective)},
                    valid=False,
                    fail_reason="empty_smiles",
                    metadata={**self.scorer_metadata(), "smiles": smiles},
                )

            value = float(self._score_fn(smiles))

            md = {**self.scorer_metadata(), "smiles": smiles}
            if context:
                # keep metadata shallow and JSON-friendly
                md["context"] = dict(context)

            return ScoringResult(
                objective=value,
                components={"objective": value},
                valid=True,
                fail_reason=None,
                metadata=md,
            )

        except Exception as e:
            md = {**self.scorer_metadata()}
            # record exception details for reproducibility/debugging
            md["exception_type"] = type(e).__name__
            md["exception_message"] = str(e)

            # Try to include SMILES (best effort)
            try:
                md["smiles"] = (
                    Chem.MolToSmiles(mol, isomericSmiles=True)
                    if mol is not None
                    else None
                )
            except Exception:
                md["smiles"] = None

            if context:
                md["context"] = dict(context)

            return ScoringResult(
                objective=float(self.fail_objective),
                components={"objective": float(self.fail_objective)},
                valid=False,
                fail_reason="legacy_score_fn_exception",
                metadata=md,
            )
