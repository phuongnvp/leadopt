from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from rdkit import Chem

Scorer = Callable[[Chem.Mol], float]


@dataclass(frozen=True)
class EpisodeResult:
    lead_smiles: str
    final_smiles: str
    lead_score: float
    final_score: float
    trajectory: List[Dict[str, Any]]  # reuse env.state.info["trajectory"] rows


class GreedyOneStepBaseline:
    """
    At each step: enumerate all valid actions, pick the action that maximizes immediate score.
    Deterministic tie-breaker: smallest action index.
    """

    def __init__(self, scorer: Scorer, max_steps: int = 12):
        self.scorer = scorer
        self.max_steps = int(max_steps)

    def run_episode(self, env, *, lead_smiles: str) -> EpisodeResult:
        env.reset(lead_smiles)

        lead_mol = env.state.mol
        lead_score = float(self.scorer(lead_mol))

        for _ in range(self.max_steps):
            actions, mask = env.enumerate_actions()
            valid_idxs = np.flatnonzero(mask)
            if valid_idxs.size == 0:
                break

            best_i: Optional[int] = None
            best_score = -1.0

            # Evaluate immediate next-state score for each valid action
            for i in map(int, valid_idxs.tolist()):
                snap = env.clone_state()
                _res = env.step(i)
                s = float(self.scorer(env.state.mol))
                env.restore_state(snap)

                if (s > best_score) or (
                    s == best_score and (best_i is None or i < best_i)
                ):
                    best_score = s
                    best_i = i

            if best_i is None:
                break

            res = env.step(best_i)
            if res.done:
                break

        final_mol = env.state.mol
        final_score = float(self.scorer(final_mol))

        return EpisodeResult(
            lead_smiles=Chem.MolToSmiles(lead_mol, canonical=True),
            final_smiles=Chem.MolToSmiles(final_mol, canonical=True),
            lead_score=lead_score,
            final_score=final_score,
            trajectory=list(env.state.info.get("trajectory", [])),
        )


class StochasticHillClimbBaseline:
    """
    Simple local search:
    - propose K random valid actions each step (deterministic RNG)
    - accept the best improving proposal (greedy hill climb)
    """

    def __init__(
        self,
        scorer: Scorer,
        max_steps: int = 12,
        proposals_per_step: int = 32,
        seed: int = 0,
    ):
        self.scorer = scorer
        self.max_steps = int(max_steps)
        self.proposals_per_step = int(proposals_per_step)
        self.rng = np.random.default_rng(seed)

    def run_episode(self, env, *, lead_smiles: str) -> EpisodeResult:
        env.reset(lead_smiles)

        lead_mol = env.state.mol
        cur_score = float(self.scorer(lead_mol))
        lead_score = cur_score

        for _ in range(self.max_steps):
            actions, mask = env.enumerate_actions()
            valid_idxs = np.flatnonzero(mask)
            if valid_idxs.size == 0:
                break

            k = min(self.proposals_per_step, int(valid_idxs.size))
            cand_idxs = self.rng.choice(valid_idxs, size=k, replace=False)

            best_i: Optional[int] = None
            best_score = cur_score

            for i in map(int, cand_idxs.tolist()):
                snap = env.clone_state()
                _res = env.step(i)
                s = float(self.scorer(env.state.mol))
                env.restore_state(snap)

                if s > best_score:
                    best_score = s
                    best_i = i

            if best_i is None:
                break  # no improvement found

            res = env.step(best_i)
            cur_score = best_score
            if res.done:
                break

        final_mol = env.state.mol
        final_score = float(self.scorer(final_mol))

        return EpisodeResult(
            lead_smiles=Chem.MolToSmiles(lead_mol, canonical=True),
            final_smiles=Chem.MolToSmiles(final_mol, canonical=True),
            lead_score=lead_score,
            final_score=final_score,
            trajectory=list(env.state.info.get("trajectory", [])),
        )
