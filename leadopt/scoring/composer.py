from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

from .types import ScoringResult

RewardMode = Literal["terminal", "delta", "potential"]


@dataclass(frozen=True)
class RewardComposer:
    """
    Converts ScoringResult(s) into an RL reward.

    Conventions:
    - objective is "higher is better"
    - constraint margins: positive = OK, negative = violation
    - penalties reduce reward (subtract from base reward)

    Parameters
    ----------
    mode:
        "terminal": reward only at terminal step, based on current objective
        "delta": reward = cur.objective - prev.objective
        "potential": reward = gamma * cur.objective - prev.objective
    gamma:
        Used only for potential-based shaping.
    step_penalty:
        Constant penalty applied each step (including terminal unless you choose otherwise).
    constraint_penalty_weight:
        Weight applied to sum of negative margin magnitudes.
        penalty = w * sum(max(0, -m) for m in constraints.values())
    compute_cost_weight:
        Weight applied to compute cost in cur.metadata["compute_cost"] (default 0).
        penalty = w * compute_cost
    bonus_weight:
        Weight applied to optional bonus term in cur.components["bonus"] (default 0).
    """

    mode: RewardMode = "delta"
    gamma: float = 0.99

    step_penalty: float = 0.0
    constraint_penalty_weight: float = 0.0
    compute_cost_weight: float = 0.0
    bonus_weight: float = 0.0
    complexity_weight: float = 0.0

    def compose(
        self,
        prev: Optional[ScoringResult],
        cur: ScoringResult,
        done: bool,
        step: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute reward and a breakdown dict for logging.

        Returns
        -------
        reward: float
        breakdown: dict with base/penalties/total and diagnostic flags
        """
        # ----- base reward -----
        if self.mode == "terminal":
            base = float(cur.objective) if done else 0.0
        elif self.mode == "delta":
            if prev is None:
                base = 0.0
            else:
                base = float(cur.objective) - float(prev.objective)
        elif self.mode == "potential":
            if prev is None:
                base = 0.0
            else:
                base = self.gamma * float(cur.objective) - float(prev.objective)
        else:
            raise ValueError(f"Unknown reward mode: {self.mode}")

        # ----- constraint penalties -----
        # Only penalize violations (negative margins)
        neg_sum = 0.0
        for _, margin in (cur.constraints or {}).items():
            try:
                m = float(margin)
            except Exception:
                # ignore malformed margins; constraints should be numeric
                continue
            if m < 0.0:
                neg_sum += -m

        constraint_penalty = self.constraint_penalty_weight * neg_sum

        # ----- compute cost penalty -----
        # Contract:
        # - prefer compute_cost_units (unitless, e.g., docking calls)
        # - fallback to legacy compute_cost
        # - last resort: compute_cost_s (wall time)
        compute_cost_units = 0.0
        compute_cost_s = None

        if cur.metadata:
            # Optional wall time (for logging; not used unless as fallback)
            if "compute_cost_s" in cur.metadata:
                try:
                    compute_cost_s = float(cur.metadata.get("compute_cost_s"))
                except Exception:
                    compute_cost_s = None

            if "compute_cost_units" in cur.metadata:
                try:
                    compute_cost_units = float(
                        cur.metadata.get("compute_cost_units", 0.0) or 0.0
                    )
                except Exception:
                    compute_cost_units = 0.0
            elif "compute_cost" in cur.metadata:
                try:
                    compute_cost_units = float(
                        cur.metadata.get("compute_cost", 0.0) or 0.0
                    )
                except Exception:
                    compute_cost_units = 0.0
            elif compute_cost_s is not None:
                # Backstop: allow using time as a cost if nothing else is provided.
                compute_cost_units = float(compute_cost_s)

        compute_cost_penalty = self.compute_cost_weight * compute_cost_units

        # ----- bonus -----
        bonus_raw = 0.0
        if cur.components and "bonus" in cur.components:
            try:
                bonus_raw = float(cur.components.get("bonus", 0.0) or 0.0)
            except Exception:
                bonus_raw = 0.0
        bonus = self.bonus_weight * bonus_raw

        # ----- step penalty -----
        step_pen = float(self.step_penalty)

        # ----- total (before shaping) -----
        reward = base + bonus - constraint_penalty - compute_cost_penalty - step_pen

        # ----- complexity shaping (Phase 7.6; optional) -----
        # Term: complexity_weight * (complexity_prev - complexity_curr)
        # Expected to be provided via `context["complexity_delta"]` by the environment.
        complexity_term = 0.0
        if float(getattr(self, "complexity_weight", 0.0)) != 0.0 and context:
            delta = context.get("complexity_delta", None)
            if delta is not None:
                try:
                    complexity_term = float(self.complexity_weight) * float(delta)
                except Exception:
                    complexity_term = 0.0

        reward = float(reward) + float(complexity_term)

        breakdown: Dict[str, Any] = {
            "mode": self.mode,
            "done": bool(done),
            "step": int(step),
            "valid": bool(cur.valid),
            "fail_reason": cur.fail_reason,
            "objective": float(cur.objective),
            "prev_objective": float(prev.objective) if prev is not None else None,
            "base_reward": float(base),
            "bonus": float(bonus),
            "constraint_violation_sum": float(neg_sum),
            "constraint_penalty": float(constraint_penalty),
            # Keep key "compute_cost" for backwards compatibility in logs;
            # it now corresponds to the units used for penalty.
            "compute_cost": float(compute_cost_units),
            "compute_cost_units": float(compute_cost_units),
            "compute_cost_s": (
                float(compute_cost_s) if compute_cost_s is not None else None
            ),
            "compute_cost_penalty": float(compute_cost_penalty),
            "step_penalty": float(step_pen),
            # Phase 7.6
            "complexity_term": float(complexity_term),
            "total_reward": float(reward),
        }

        # Pass through any optional context keys (useful later for budgets)
        if context:
            # shallow copy only; keep breakdown JSON-friendly
            for k, v in context.items():
                if k not in breakdown:
                    breakdown[k] = v

        return reward, breakdown
