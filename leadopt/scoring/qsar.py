from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, rdMolDescriptors

from .base import Scorer
from .types import ScoringResult


@dataclass
class QSARScorer(Scorer):
    """Deterministic QSAR-style scorer (placeholder for academic workflows).

    This scorer is intended as the first "real" Scorer implementation under
    ``leadopt.scoring``. It provides a cheap, deterministic objective suitable
    for validating end-to-end RL + YAML integration.

    Parameters
    ----------
    objective:
        Which property to use as the main objective. Currently supported:

        - ``"qed"``: RDKit QED (0..1, higher is better)
        - ``"mw_desirability"``: triangular desirability around ``mw_target``

    mw_target:
        Target MW for ``mw_desirability``.
    mw_tolerance:
        Half-width around ``mw_target`` where desirability declines linearly to 0.
    fail_objective:
        Objective value to use on failure (must be very low). If not provided,
        uses the base ``Scorer.fail_objective``.
    version:
        Stable version string for manifests.
    extra_metadata:
        Extra fields to include in ``scorer_metadata()``.
    """

    objective: str = "qed"
    mw_target: float = 350.0
    mw_tolerance: float = 150.0

    # Allow YAML override while keeping the Scorer base default.
    fail_objective: float = Scorer.fail_objective

    # Stable version string for reproducibility artifacts.
    version: str = "0"

    extra_metadata: Optional[Dict[str, Any]] = None

    def scorer_metadata(self) -> Dict[str, Any]:
        md: Dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "type": "qsar",
            "objective": self.objective,
            "mw_target": float(self.mw_target),
            "mw_tolerance": float(self.mw_tolerance),
        }
        if self.extra_metadata:
            md.update(dict(self.extra_metadata))
        return md

    @staticmethod
    def _mw_desirability(mw: float, target: float, tol: float) -> float:
        """Triangular desirability in [0, 1] centered at target."""
        tol = max(float(tol), 1e-12)
        d = abs(float(mw) - float(target))
        return max(0.0, 1.0 - (d / tol))

    def score(
        self, mol: Any, context: Optional[Dict[str, Any]] = None
    ) -> ScoringResult:
        try:
            if mol is None:
                return ScoringResult(
                    objective=float(self.fail_objective),
                    components={"objective": float(self.fail_objective)},
                    valid=False,
                    fail_reason="input:mol_is_none",
                    metadata={**self.scorer_metadata(), "exception_type": None},
                )

            # Canonical SMILES for reproducible logging.
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            if not smiles:
                return ScoringResult(
                    objective=float(self.fail_objective),
                    components={"objective": float(self.fail_objective)},
                    valid=False,
                    fail_reason="input:empty_smiles",
                    metadata={**self.scorer_metadata(), "smiles": smiles},
                )

            # Deterministic RDKit properties.
            mw = float(Descriptors.MolWt(mol))
            logp = float(Crippen.MolLogP(mol))
            tpsa = float(rdMolDescriptors.CalcTPSA(mol))
            qed = float(QED.qed(mol))

            obj_key = str(self.objective).strip().lower()
            if obj_key == "qed":
                objective = qed
            elif obj_key == "mw_desirability":
                objective = float(
                    self._mw_desirability(mw, self.mw_target, self.mw_tolerance)
                )
            else:
                return ScoringResult(
                    objective=float(self.fail_objective),
                    components={"objective": float(self.fail_objective)},
                    valid=False,
                    fail_reason=f"input:unknown_objective:{self.objective}",
                    metadata={**self.scorer_metadata(), "smiles": smiles},
                )

            md = {**self.scorer_metadata(), "smiles": smiles}
            if context:
                # Keep shallow and JSON-friendly.
                md["context"] = dict(context)

            # Floats-only components (reproducibility).
            components: Dict[str, float] = {
                "objective": float(objective),
                "qed": float(qed),
                "mw": float(mw),
                "logp": float(logp),
                "tpsa": float(tpsa),
            }

            return ScoringResult(
                objective=float(objective),
                components=components,
                valid=True,
                fail_reason=None,
                metadata=md,
            )

        except Exception as e:
            md = {**self.scorer_metadata()}
            md["exception_type"] = type(e).__name__
            md["exception_message"] = str(e)
            # Best-effort SMILES.
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
                fail_reason=f"exception:{type(e).__name__}",
                metadata=md,
            )
