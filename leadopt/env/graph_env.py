# leadopt/env/graph_env.py
import copy
import json
import math
import random
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem

from leadopt.core.complexity import compute_complexity_score

from ..actions import ActionInstance, ActionOperator, ActionSpace, AppliedAction
from ..constraints import Constraint, ConstraintSuite

#from ..constraints.suite import ConstraintSuite
from ..core.errors import ActionError
from ..core.mol import MoleculeState
from ..core.rdkit_utils import assert_valid_mol, canonical_smiles
from ..core.rules import RuleConfig
from ..scoring.base import Scorer
from ..scoring.composer import RewardComposer
from ..scoring.legacy import LegacyFunctionScorer

ScoreFn = Callable[[str], float]


@dataclass(frozen=True)
class StepResult:
    state: MoleculeState
    reward: float
    done: bool
    info: Dict[str, object]


@dataclass
class EnvSnapshot:
    """
    Full environment snapshot for "simulate action then revert" baselines.
    Deterministic deep copy (RDKit Mol copies + deep-copied info/trajectory).
    """
    lead: Optional[Chem.Mol]
    state: Optional[MoleculeState]
    done: bool
    cached_actions: List[ActionInstance]
    cached_mask: np.ndarray
    rng_state: Dict[str, object]


class TerminateOperator(ActionOperator):
    """Pseudo-operator: provides a TERMINATE action that ends the episode immediately."""
    name = "Terminate"

    def enumerate_actions(self, mol: Chem.Mol, ctx) -> Sequence[ActionInstance]:
        return [ActionInstance(operator=self.name, site=tuple(), template=None, payload={"terminate": True})]

    def touched(self, mol: Chem.Mol, action: ActionInstance) -> tuple[set[int], set[int]]:
        self._ensure_operator_match(action)
        return set(), set()

    def apply(self, mol: Chem.Mol, action: ActionInstance) -> AppliedAction:
        self._ensure_operator_match(action)
        m = self._prepare_working_copy(mol)
        return AppliedAction(mol=m, action=action, touched_atoms=set(), touched_bonds=set())


class GraphEnvironment:

    def __init__(
        self,
        *,
        operators: Sequence[ActionOperator],
        scorer: Optional[Scorer] = None,
        reward_composer: Optional[RewardComposer] = None,
        score_fn: Optional[ScoreFn] = None,
        max_steps: int = 8,
        seed: int = 0,
        require_connected: bool = True,
        include_terminate: bool = True,
        constraint_suite: Optional[ConstraintSuite] = None,          # NEW
        constraint_factory: Optional[Callable[[], Constraint]] = None,
        rule_config: Optional[RuleConfig] = None,
        reward_mode: str = "potential",
        gamma: float = 0.99,
        step_penalty: float = 0.0,
    ) -> None:

        self.max_steps = int(max_steps)
        self.require_connected = require_connected


        self._base_operators: List[ActionOperator] = list(operators)
        self._include_terminate: bool = bool(include_terminate)

        self.constraint_suite = constraint_suite
        self.constraint_factory = constraint_factory
        self.rule_config = rule_config

        self.rng = np.random.default_rng(seed)
        # --- Hybrid init-pool support (opt-in, backward compatible) ---
        self._init_smiles_pool: Optional[list[str]] = None
        self._init_pool_scores: Optional[list[float]] = None  # optional for softmax sampling
        self._init_pool_sampling: str = "uniform"  # "uniform" | "softmax"
        self._init_pool_temperature: float = 1.0
        # Deterministic RNG for pool sampling; can be reseeded externally if you already have seeding infra.
        self._init_rng = random.Random(0)

        # Scorer wiring:
        # - Prefer explicit scorer
        # - Else wrap legacy score_fn
        if scorer is not None:
            self.scorer: Scorer = scorer
        else:
            if score_fn is None:
                raise ValueError("GraphEnvironment requires either scorer=... or score_fn=...")
            self.scorer = LegacyFunctionScorer(score_fn)

        # Reward composer wiring:
        # - Prefer explicit reward_composer
        # - Else build one from legacy args (so existing scripts keep working)
        if reward_composer is not None:
            self.reward_composer: RewardComposer = reward_composer
        else:
            self.reward_composer = RewardComposer(
                mode=str(reward_mode),
                gamma=float(gamma),
                step_penalty=float(step_penalty),
            )

        self._lead: Optional[Chem.Mol] = None
        self._state: Optional[MoleculeState] = None
        self._done: bool = True

        # Action space is per-episode because constraint is per-episode
        self._action_space: Optional[ActionSpace] = None

        # cached per-state action list/mask (for reproducibility within a step)
        self._cached_actions: List[ActionInstance] = []
        self._cached_mask = np.zeros((0,), dtype=bool)
        self._cached_key: tuple[int, str] | None = None
        self._cached_applied = None  # list[Optional[Chem.Mol]] after your caching patch


    def _payload_key(self, payload: dict) -> str:
        return json.dumps(payload or {}, sort_keys=True, separators=(",", ":"))
    
    @property
    def state(self) -> MoleculeState:
        if self._state is None:
            raise RuntimeError("Environment not reset.")
        return self._state

    @property
    def done(self) -> bool:
        return self._done
    
    def _eval_constraints(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Evaluate reward/logging constraints (NOT action legality constraints).
        Failure-safe: returns {} if no suite.
        """
        if self.constraint_suite is None:
            return {}
        try:
            return self.constraint_suite.evaluate_all(mol, context=None)
        except Exception:
            # Suite itself should be failure-safe already, but keep env robust.
            return {}

    # ------------------------------------------------------------------
    # Snapshot support (for baselines / action lookahead evaluation)
    # ------------------------------------------------------------------

    def clone_state(self) -> EnvSnapshot:
        """
        Deep snapshot of the environment state. Safe to use for:
        - try action
        - score
        - revert
        """
        lead_copy = Chem.Mol(self._lead) if self._lead is not None else None

        state_copy: Optional[MoleculeState]
        if self._state is None:
            state_copy = None
        else:
            state_copy = MoleculeState(
                mol=Chem.Mol(self._state.mol),
                step=int(self._state.step),
                max_steps=int(self._state.max_steps),
                info=copy.deepcopy(self._state.info),
            )

        # ActionInstances are typically small dataclasses; deepcopy is safe.
        cached_actions_copy = copy.deepcopy(self._cached_actions)
        cached_mask_copy = self._cached_mask.copy()

        # Preserve RNG state too (even though step() doesn't use it, this avoids surprises)
        rng_state_copy: Dict[str, object] = copy.deepcopy(self.rng.bit_generator.state)

        return EnvSnapshot(
            lead=lead_copy,
            state=state_copy,
            done=bool(self._done),
            cached_actions=cached_actions_copy,
            cached_mask=cached_mask_copy,
            rng_state=rng_state_copy,
        )

    def restore_state(self, snap: EnvSnapshot) -> None:
        """
        Restore a previously captured snapshot. Does not rebuild the ActionSpace,
        because ActionSpace is per-episode and should remain identical during lookahead.
        """
        self._lead = Chem.Mol(snap.lead) if snap.lead is not None else None

        if snap.state is None:
            self._state = None
        else:
            self._state = MoleculeState(
                mol=Chem.Mol(snap.state.mol),
                step=int(snap.state.step),
                max_steps=int(snap.state.max_steps),
                info=copy.deepcopy(snap.state.info),
            )

        self._done = bool(snap.done)

        self._cached_actions = copy.deepcopy(snap.cached_actions)
        self._cached_mask = snap.cached_mask.copy()

        # Restore RNG state
        self.rng.bit_generator.state = copy.deepcopy(snap.rng_state)

    # ------------------------------------------------------------------
    # Core env API
    # ------------------------------------------------------------------
    def reset(self, lead_smiles: str) -> MoleculeState:
        st = MoleculeState.from_smiles(
            lead_smiles,
            step=0,
            max_steps=self.max_steps,
        )
        assert_valid_mol(st.mol, require_connected=self.require_connected)

        # Cache initial ScoringResult for shaping/logging
        st.info = dict(getattr(st, "info", {}) or {})
        result0 = self.scorer.score(st.mol, context=None)
        margins0 = self._eval_constraints(st.mol)
        st.info["_result"] = result0.with_constraints(margins0)

        self._lead = Chem.Mol(st.mol)
        self._state = st
        self._done = False
        self._cached_key = None

        constraint = self.constraint_factory() if self.constraint_factory is not None else None
        ops = list(self._base_operators)
        if self._include_terminate:
            ops.append(TerminateOperator())
        self._action_space = ActionSpace(
            operators=ops,
            constraint=constraint,
            rule_config=self.rule_config,
        )

        self._refresh_action_cache()
        # Ensure cache key is aligned with the just-built cache to avoid redundant refresh.
        self._cached_key = (int(self._state.step), canonical_smiles(self._state.mol))
        return self._state


    def set_initial_smiles_pool(
        self,
        smiles: Sequence[str],
        *,
        scores: Optional[Sequence[float]] = None,
        sampling: str = "uniform",
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Configure an optional initial-state pool for hybrid beam->RL.

        - sampling="uniform": uniform random from pool
        - sampling="softmax": sample proportional to exp(score / temperature)
        (scores must be provided; higher score => more likely)

        Backwards compatible: if this is never called, nothing changes.
        """
        if not smiles:
            raise ValueError("Initial SMILES pool must be non-empty.")
        self._init_smiles_pool = [str(s) for s in smiles]

        if scores is not None:
            if len(scores) != len(smiles):
                raise ValueError("scores must have same length as smiles.")
            self._init_pool_scores = [float(x) for x in scores]
        else:
            self._init_pool_scores = None

        sampling = str(sampling)
        if sampling not in ("uniform", "softmax"):
            raise ValueError("sampling must be 'uniform' or 'softmax'")
        if sampling == "softmax" and self._init_pool_scores is None:
            raise ValueError("sampling='softmax' requires scores")

        self._init_pool_sampling = sampling
        self._init_pool_temperature = float(temperature)
        if self._init_pool_temperature <= 0:
            raise ValueError("temperature must be > 0")

        if seed is not None:
            self._init_rng.seed(int(seed))


    def reset_from_pool(self) -> "MoleculeState":
        """
        Reset environment by sampling an initial SMILES from the configured pool.

        Deterministic given the pool + seed passed to set_initial_smiles_pool().
        """
        if not self._init_smiles_pool:
            raise ValueError("Initial SMILES pool not configured. Call set_initial_smiles_pool(...) first.")

        if self._init_pool_sampling == "uniform":
            smi = self._init_rng.choice(self._init_smiles_pool)
            return self.reset(smi)

        # softmax sampling
        assert self._init_pool_scores is not None
        t = float(self._init_pool_temperature)
        # Numerically stable softmax: subtract max
        m = max(self._init_pool_scores)
        weights = [math.exp((s - m) / t) for s in self._init_pool_scores]
        total = sum(weights)
        if total <= 0:
            # fallback to uniform
            smi = self._init_rng.choice(self._init_smiles_pool)
            return self.reset(smi)

        r = self._init_rng.random() * total
        acc = 0.0
        for smi, w in zip(self._init_smiles_pool, weights):
            acc += w
            if acc >= r:
                return self.reset(smi)

        # numeric fallback
        return self.reset(self._init_smiles_pool[-1])

    def _refresh_action_cache(self) -> None:
        """
        Refresh the list of available actions, a boolean mask of which are allowed,
        AND a cache of the resulting molecules for each allowed action.

        This prevents "double apply" (apply once for masking, again for stepping)
        and eliminates mask/step inconsistency due to RDKit edge cases.
        """
        if self._state is None or self._action_space is None:
            self._cached_actions = []
            self._cached_mask = np.zeros((0,), dtype=bool)
            self._cached_applied = []  # list[Optional[Chem.Mol]]
            return

        mol = self._state.mol
        all_actions = self._action_space.enumerate(mol)

        # NEW: get both allowed actions and applied mols aligned with all_actions
        allowed_actions, applied_mols = self._action_space.filter_allowed_with_applied(mol, all_actions)

        self._cached_actions = list(all_actions)

        # applied_mols is aligned with all_actions; entries are Chem.Mol for allowed else None
        self._cached_applied = list(applied_mols)

        # Mask consistency rule: allowed iff applied_mols[i] is not None
        self._cached_mask = np.array([m is not None for m in self._cached_applied], dtype=bool)

    def available_actions(self) -> Tuple[List[ActionInstance], np.ndarray]:
        """
        Returns (actions, mask) where mask[i]==True means action i is allowed.

        Cached per-state (step + canonical smiles) to keep sampling deterministic
        within a step while avoiding stale actions after molecular transitions.

        Phase 7.6 contract: mask must reflect ActionSpace gating (rules/constraints),
        and must be consistent with stepping.
        """
        if self._state is None or self._action_space is None:
            raise RuntimeError("Environment not reset.")

        key = (int(self._state.step), canonical_smiles(self._state.mol))
        if self._cached_key != key:
            self._refresh_action_cache()
            self._cached_key = key

        return self._cached_actions, self._cached_mask.copy()



    # Compatibility alias used by baseline/evaluation utilities
    def enumerate_actions(self) -> Tuple[List[ActionInstance], np.ndarray]:
        return self.available_actions()

    def step(self, action_index: int) -> StepResult:
        if self._state is None or self._action_space is None:
            raise RuntimeError("Environment not reset.")
        if self._done:
            raise RuntimeError("Episode already done. Call reset().")

        actions, mask = self.available_actions()

        if action_index < 0 or action_index >= len(actions):
            raise IndexError("action_index out of range.")
        if not bool(mask[action_index]):
            raise ActionError("Chosen action is masked (not allowed).")

        action = actions[action_index]
        mol = self._state.mol

        # Ensure result cache exists for current state
        prev_result = self._state.info.get("_result", None)
        if prev_result is None:
            prev_result = self.scorer.score(mol, context=None).with_constraints(self._eval_constraints(mol))
            self._state.info["_result"] = prev_result
        else:
            # If older state had no constraints (e.g., loading old checkpoints), backfill once.
            if not getattr(prev_result, "constraints", None):
                prev_result = prev_result.with_constraints(self._eval_constraints(mol))
                self._state.info["_result"] = prev_result

        # -------------------------------
        # Phase 7.6: complexity (prev)
        # -------------------------------
        try:
            complexity_prev = float(compute_complexity_score(mol))
        except Exception:
            complexity_prev = None

        # Termination action
        if action.operator == "Terminate":
            self._done = True

            # No molecular transition; we compose reward from the current result.
            # We explicitly avoid the (gamma-1)*objective artifact in potential shaping on "no transition".
            if self.reward_composer.mode == "terminal":
                reward, breakdown = self.reward_composer.compose(
                    prev=prev_result,
                    cur=prev_result,
                    done=True,
                    step=int(self._state.step),
                    context={
                        "terminated_by": "terminate_action",
                        # Include complexity metadata when available
                        "complexity_prev": complexity_prev,
                        "complexity_curr": complexity_prev,
                        "complexity_delta": 0.0 if complexity_prev is not None else None,
                    },
                )
            else:
                # delta/potential: no transition -> reward should be 0 (except step_penalty handled by composer)
                # To preserve that behavior, we force base=0 by passing prev=None.
                reward, breakdown = self.reward_composer.compose(
                    prev=None,
                    cur=prev_result,
                    done=True,
                    step=int(self._state.step),
                    context={
                        "terminated_by": "terminate_action",
                        "no_transition": True,
                        "complexity_prev": complexity_prev,
                        "complexity_curr": complexity_prev,
                        "complexity_delta": 0.0 if complexity_prev is not None else None,
                    },
                )

            info = {
                "terminated": True,
                "reason": "terminate_action",
                "smiles": canonical_smiles(mol),
                "reward_breakdown": breakdown,
            }

            # Mirror complexity metadata into info/state if available
            if complexity_prev is not None:
                info["complexity_prev"] = complexity_prev
                info["complexity_curr"] = complexity_prev
                info["complexity_delta"] = 0.0
                self._state.info["complexity_prev"] = complexity_prev
                self._state.info["complexity_curr"] = complexity_prev
                self._state.info["complexity_delta"] = 0.0

            return StepResult(state=self._state, reward=float(reward), done=True, info=info)

        # -----------------------------------------
        # Non-terminate: transition -> new mol
        # -----------------------------------------
        # Phase 7.6 contract: step execution must be consistent with the mask produced by
        # ActionSpace.filter_allowed_with_applied(). Prefer reusing cached applied mols to
        # avoid double-apply divergence (RDKit edge cases).
        new_mol: Optional[Chem.Mol] = None

        if self._cached_applied is not None:
            try:
                new_mol = self._cached_applied[action_index]
            except Exception:
                new_mol = None

        if new_mol is None:
            # Backward-compatible fallback: apply operator directly (should be rare).
            op_map = {op.name: op for op in self._action_space.operators}
            if action.operator not in op_map:
                raise RuntimeError(f"Unknown operator in action: {action.operator!r}")
            op = op_map[action.operator]
            applied = op.apply(mol, action)  # may raise ActionError (intended)
            new_mol = applied.mol

        assert_valid_mol(new_mol, require_connected=self.require_connected)

        # -------------------------------
        # Phase 7.6: complexity (curr)
        # -------------------------------
        try:
            complexity_curr = float(compute_complexity_score(new_mol))
            complexity_delta = float(complexity_prev - complexity_curr) if complexity_prev is not None else None
        except Exception:
            complexity_curr = None
            complexity_delta = None

        # Score next state and cache result in next_state.info
        cur_result = self.scorer.score(new_mol, context=None).with_constraints(self._eval_constraints(new_mol))

        # Step counter update (done if max_steps reached)
        next_step = int(self._state.step) + 1
        done = bool(next_step >= int(self.max_steps))

        # Compose reward
        reward, breakdown = self.reward_composer.compose(
            prev=prev_result,
            cur=cur_result,
            done=done,
            step=int(self._state.step),
            context={
                # Provide transition metadata for optional shaping
                "action_operator": action.operator,
                "complexity_prev": complexity_prev,
                "complexity_curr": complexity_curr,
                "complexity_delta": complexity_delta,
            },
        )

        # Build next state (do NOT mutate old mol)
        next_info = dict(self._state.info)
        next_info["_result"] = cur_result

        # Mirror complexity metadata into next_state.info
        if complexity_prev is not None:
            next_info["complexity_prev"] = complexity_prev
        if complexity_curr is not None:
            next_info["complexity_curr"] = complexity_curr
        if complexity_delta is not None:
            next_info["complexity_delta"] = complexity_delta

        next_state = replace(self._state, mol=new_mol, step=next_step, info=next_info)

        # Update env state + done flag
        self._state = next_state
        self._cached_key = None
        self._done = done

        # Step info (also mirror complexity here)
        info = {
            "terminated": False,
            "smiles": canonical_smiles(new_mol),
            "reward_breakdown": breakdown,
        }
        if complexity_prev is not None:
            info["complexity_prev"] = complexity_prev
        if complexity_curr is not None:
            info["complexity_curr"] = complexity_curr
        if complexity_delta is not None:
            info["complexity_delta"] = complexity_delta

        return StepResult(state=next_state, reward=float(reward), done=done, info=info)

    # ---------------------------
    # Dummy policy helpers
    # ---------------------------

    def sample_random_action_index(self) -> int:
        """Uniformly sample among allowed actions for the current state."""
        _, mask = self.available_actions()
        allowed = np.flatnonzero(mask)
        if allowed.size == 0:
            raise ActionError("No valid actions available.")
        return int(self.rng.choice(allowed))

    def rollout_random(self, *, max_env_steps: Optional[int] = None) -> StepResult:
        """
        Rollout using the dummy random policy until done or until max_env_steps steps.
        Returns the final StepResult (terminal if reached).
        """
        if self._state is None:
            raise RuntimeError("Environment not reset.")

        limit = int(max_env_steps if max_env_steps is not None else (self.max_steps + 5))
        last: Optional[StepResult] = None

        for _ in range(limit):
            if self.done:
                break
            aidx = self.sample_random_action_index()
            last = self.step(aidx)

        if last is None:
            return StepResult(state=self.state, reward=0.0, done=self.done, info={"terminated": self.done})
        return last