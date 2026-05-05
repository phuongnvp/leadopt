from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem

from leadopt.actions.space import ActionSpace
from leadopt.constraints.suite import ConstraintSuite
from leadopt.core.complexity import compute_complexity_score
from leadopt.core.rdkit_utils import assert_valid_mol
from leadopt.scoring.base import Scorer


@dataclass(frozen=True)
class BeamItem:
    """A single beam-search state."""

    smiles: str
    objective: float
    complexity: float
    augmented_objective: float
    step: int

    # Lightweight provenance
    parent_smiles: Optional[str]
    action_operator: Optional[str]
    action_template: Optional[str]

    # Optional diagnostics
    constraints: Dict[str, float]
    metadata: Dict[str, Any]


def _canonical_smiles(m: Chem.Mol) -> str:
    return Chem.MolToSmiles(m, canonical=True)


def _score_with_constraints(
    mol: Chem.Mol,
    *,
    scorer: Scorer,
    constraint_suite: Optional[ConstraintSuite],
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    """Return (objective, margins, metadata). Objective is higher-is-better."""
    res = scorer.score(mol, context=context)
    objective = float(res.objective)
    margins: Dict[str, float] = {}
    if constraint_suite is not None:
        margins = constraint_suite.evaluate_all(mol, context=context)
    meta = dict(res.metadata or {})
    # Keep scorer failure info in metadata
    meta["scorer_valid"] = bool(res.valid)
    meta["scorer_fail_reason"] = res.fail_reason
    return objective, margins, meta


def beam_decomplexify(
    *,
    start_smiles: str,
    action_space: ActionSpace,
    scorer: Scorer,
    constraint_suite: Optional[ConstraintSuite] = None,
    beam_width: int = 20,
    max_steps: int = 8,
    per_state_action_limit: int = 256,
    complexity_weight: float = 0.0,
    docking_drop_tolerance: Optional[float] = None,
    hard_constraint_filter: bool = True,
    context: Optional[Dict[str, Any]] = None,
) -> List[BeamItem]:
    """Deterministic beam search for decomplexification.

    Parameters
    ----------
    start_smiles:
        Starting ligand SMILES.
    action_space:
        ActionSpace configured with operators, legality constraint, and rules.
    scorer:
        Scorer (e.g., DockingScorer). Higher objective is better.
    constraint_suite:
        Optional ConstraintSuite for margins. Used for filtering/logging.
    beam_width:
        Number of states kept per depth.
    max_steps:
        Search depth.
    per_state_action_limit:
        Cap actions expanded per state (after deterministic ordering).
    complexity_weight:
        Augmented objective = objective - complexity_weight * complexity.
        Use >0 to encourage simplification.
    docking_drop_tolerance:
        Optional: require objective >= (start_objective - docking_drop_tolerance).
        This enforces "similar activity" as docking retention.
    hard_constraint_filter:
        If True, reject any candidate with any margin < 0.
    context:
        Optional dict passed to scorer/constraints.

    Returns
    -------
    items:
        All accepted BeamItems encountered (including start), sorted by augmented_objective desc.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if max_steps < 0:
        raise ValueError("max_steps must be >= 0")
    if per_state_action_limit < 1:
        raise ValueError("per_state_action_limit must be >= 1")

    start_mol = Chem.MolFromSmiles(start_smiles)
    if start_mol is None:
        raise ValueError("start_smiles could not be parsed by RDKit")
    assert_valid_mol(start_mol)
    start_smi = _canonical_smiles(start_mol)

    start_obj, start_margins, start_meta = _score_with_constraints(
        start_mol, scorer=scorer, constraint_suite=constraint_suite, context=context
    )
    start_cx = float(compute_complexity_score(start_mol))
    start_aug = float(start_obj) - float(complexity_weight) * float(start_cx)

    if docking_drop_tolerance is not None:
        # Start is always allowed; threshold is defined relative to start objective.
        min_obj = float(start_obj) - float(docking_drop_tolerance)
    else:
        min_obj = None

    def _constraints_ok(margins: Dict[str, float]) -> bool:
        if not hard_constraint_filter:
            return True
        return all(float(m) >= 0.0 for m in margins.values())

    seen: set[str] = {start_smi}
    all_items: List[BeamItem] = []

    start_item = BeamItem(
        smiles=start_smi,
        objective=float(start_obj),
        complexity=float(start_cx),
        augmented_objective=float(start_aug),
        step=0,
        parent_smiles=None,
        action_operator=None,
        action_template=None,
        constraints=dict(start_margins),
        metadata=dict(start_meta),
    )
    all_items.append(start_item)

    beam: List[BeamItem] = [start_item]

    # Deterministic per-depth expansion.
    for step in range(1, int(max_steps) + 1):
        candidates: List[BeamItem] = []
        for item in beam:
            mol = Chem.MolFromSmiles(item.smiles)
            if mol is None:
                continue
            # ActionSpace enumeration is deterministic and includes legality/rule gating.
            actions = action_space.enumerate(mol)
            # Apply+rule gate once, reuse resulting mols.
            allowed, applied = action_space.filter_allowed_with_applied(mol, actions)

            # deterministic truncation to keep expansion bounded
            allowed = allowed[: int(per_state_action_limit)]
            applied = applied[: int(per_state_action_limit)]

            for a, next_mol in zip(allowed, applied):
                if next_mol is None:
                    continue
                try:
                    assert_valid_mol(next_mol)
                except Exception:
                    continue

                smi = _canonical_smiles(next_mol)
                if smi in seen:
                    continue

                obj, margins, meta = _score_with_constraints(
                    next_mol,
                    scorer=scorer,
                    constraint_suite=constraint_suite,
                    context=context,
                )
                if min_obj is not None and float(obj) < float(min_obj):
                    continue
                if not _constraints_ok(margins):
                    continue

                cx = float(compute_complexity_score(next_mol))
                aug = float(obj) - float(complexity_weight) * float(cx)

                candidates.append(
                    BeamItem(
                        smiles=smi,
                        objective=float(obj),
                        complexity=float(cx),
                        augmented_objective=float(aug),
                        step=int(step),
                        parent_smiles=item.smiles,
                        action_operator=str(a.operator),
                        action_template=str(a.template),
                        constraints=dict(margins),
                        metadata=dict(meta),
                    )
                )
                seen.add(smi)

        # Keep best candidates by augmented objective.
        candidates.sort(
            key=lambda x: (-float(x.augmented_objective), float(x.complexity), x.smiles)
        )
        beam = candidates[: int(beam_width)]
        all_items.extend(beam)

        if not beam:
            break

    all_items.sort(
        key=lambda x: (-float(x.augmented_objective), float(x.complexity), x.smiles)
    )
    return all_items
