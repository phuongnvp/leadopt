from __future__ import annotations

"""Shared implementation for `leadopt run` and `leadopt.api.run`.

This engine mirrors the current CLI behavior (random rollout) but returns
stable API objects.

Reproducibility policy (Phase 2.3):
- Seed plumbing is explicit.
- No file I/O in the engine unless requested by caller.
"""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from leadopt.api.types import (
    ActionStep,
    ActionTrace,
    MoleculeRecord,
    RunMetadata,
    RunResult,
)
from leadopt.core.seeding import set_global_seed


def _require_rdkit() -> None:
    # Backwards-compatible alias (engine-local).
    from leadopt.core import _require_rdkit as _core_require_rdkit

    _core_require_rdkit()


def _canonicalize(smiles: str) -> str:
    from leadopt.core.smiles import canonicalize_smiles_or_empty

    return canonicalize_smiles_or_empty(smiles)


def _as_dict(obj: Any) -> Dict[str, Any]:
    """Normalize rollout results to a plain dict.

    Supports:
      - Mapping (dict-like)
      - dataclasses
      - pydantic v2 (model_dump)
      - pydantic v1 (dict)
      - plain objects with __dict__
    """
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        return dict(obj.model_dump())
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        return dict(obj.dict())
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {}


def _score_from_env_scorer(
    env: Any, smiles: str
) -> tuple[float, Optional[dict[str, float]], dict[str, Any]]:
    """Compute (objective, components, metadata) from preset scorer if available.

    Best-effort: never raises.
    """
    try:
        _require_rdkit()
        from rdkit import Chem

        from leadopt.api.scoring import score_to_fields

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (
                0.0,
                None,
                {
                    "valid": False,
                    "fail_reason": "invalid_smiles",
                    "constraints": {},
                    "metadata": {},
                },
            )

        scorer = getattr(env, "scorer", None)
        if scorer is None:
            return (
                0.0,
                None,
                {
                    "valid": True,
                    "fail_reason": None,
                    "constraints": {},
                    "metadata": {"note": "no_scorer"},
                },
            )

        res = scorer.score(mol, context=None)
        return score_to_fields(res)
    except Exception:
        return 0.0, None, {}


def _iter_action_steps(actions: Any, *, fallback_smiles: str) -> Iterable[ActionStep]:
    """Best-effort conversion of an action trace into ActionStep objects."""
    if actions is None:
        return []

    steps: list[ActionStep] = []
    if isinstance(actions, list):
        for t, a in enumerate(actions):
            if isinstance(a, dict):
                op = (
                    a.get("operator")
                    or a.get("action_operator")
                    or a.get("op")
                    or "unknown"
                )
                templ = a.get("template") or a.get("action_template")
                aidx = a.get("action_index")
                inter = (
                    a.get("intermediate_smiles")
                    or a.get("smiles")
                    or a.get("state_smiles")
                    or fallback_smiles
                )
                steps.append(
                    ActionStep(
                        t=int(a.get("t", t)),
                        operator=str(op),
                        template=str(templ) if templ is not None else None,
                        action_index=int(aidx) if aidx is not None else None,
                        intermediate_smiles=str(inter),
                    )
                )
            else:
                # unknown action format; preserve as placeholder
                steps.append(
                    ActionStep(
                        t=t,
                        operator=str(a) if a is not None else "unknown",
                        template=None,
                        action_index=None,
                        intermediate_smiles=fallback_smiles,
                    )
                )
    return steps


def run_once(
    *,
    preset_path: Path,
    smiles: str,
    steps: int,
    seed: int,
    device: Optional[str] = None,
    preset_name: Optional[str] = None,
    run_dir: Optional[Path] = None,
    write_artifacts: bool = True,
) -> RunResult:
    """Run one random rollout under a preset.

    Mirrors `leadopt run` behavior (random rollout), but returns a stable RunResult.
    """
    _require_rdkit()

    from leadopt.config.preset_loader import PresetLoader

    loader = PresetLoader()
    loaded = loader.load(preset_path)

    # Unified seed plumbing
    set_global_seed(int(seed), deterministic_torch=True)

    env = loader.build_env(loaded)
    lead_can = _canonicalize(str(smiles))
    env.reset(lead_can)

    final_obj = env.rollout_random(max_env_steps=int(steps))
    final = _as_dict(final_obj)

    def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
        return getattr(obj, name, default) if obj is not None else default

    state_obj = final.get("state", None)
    if state_obj is None:
        state_obj = _get_attr(final_obj, "state", None)

    info_obj = final.get("info", None)
    if info_obj is None:
        info_obj = _get_attr(final_obj, "info", None)
    info: Dict[str, Any] = info_obj if isinstance(info_obj, dict) else {}

    # Extract final SMILES (same heuristics as CLI)
    final_smiles = None
    if isinstance(final, dict):
        final_smiles = final.get("smiles", None)
    if final_smiles is None:
        final_smiles = info.get("smiles", None) or info.get("final_smiles", None)
    if final_smiles is None and state_obj is not None:
        for attr in ("smiles", "current_smiles", "mol_smiles"):
            v = _get_attr(state_obj, attr, None)
            if v:
                final_smiles = v
                break
        if final_smiles is None and isinstance(state_obj, dict):
            final_smiles = (
                state_obj.get("smiles")
                or state_obj.get("current_smiles")
                or state_obj.get("mol_smiles")
            )
        if final_smiles is None:
            mol = _get_attr(state_obj, "mol", None) or _get_attr(
                state_obj, "rdmol", None
            )
            if mol is not None:
                try:
                    from rdkit import Chem

                    final_smiles = Chem.MolToSmiles(mol, canonical=False)
                except Exception:
                    pass

    if final_smiles is None:
        raise RuntimeError(
            "Could not extract final SMILES from rollout result. "
            f"type={type(final_obj)} keys={list(final.keys())} "
            f"has_state={state_obj is not None} has_info={bool(info)} info_keys={list(info.keys())}"
        )

    final_can = _canonicalize(str(final_smiles))

    # Extract scalar reward/score (same heuristics as CLI)
    score = None
    if isinstance(final, dict):
        score = final.get("score", None)
    if score is None:
        reward = final.get("reward", None)
        if reward is None:
            reward = _get_attr(final_obj, "reward", None)
        score = reward
    if score is None:
        score = info.get("score", None)
    score_f = float(score) if score is not None else 0.0

    actions = None
    if isinstance(final, dict):
        actions = final.get("actions", None)
    if actions is None:
        actions = info.get("actions", None) or info.get("action_trace", None)
    if actions is None and state_obj is not None:
        actions = _get_attr(state_obj, "actions", None) or _get_attr(
            state_obj, "action_trace", None
        )

    trace_steps = list(
        _iter_action_steps(actions, fallback_smiles=final_can or str(final_smiles))
    )

    lead_obj, lead_components, lead_score_md = _score_from_env_scorer(env, lead_can)
    final_obj_sc, final_components, final_score_md = (
        _score_from_env_scorer(env, final_can) if final_can else (0.0, None, {})
    )

    md = RunMetadata(
        seed=int(seed),
        device=device,
        preset_name=preset_name,
        preset_path=str(preset_path),
        run_dir=None,
        extra={
            "cli_compat": {
                "final_score": score_f,
                "final_smiles_raw": str(final_smiles),
                "actions_raw": (
                    json.loads(json.dumps(actions, default=str))
                    if actions is not None
                    else []
                ),
            }
        },
    )

    # Optional CLI-compatible artifact writing: result.json
    if run_dir is not None and bool(write_artifacts):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "result.json"
        out_path.write_text(
            json.dumps(
                {
                    "seed": int(md.seed),
                    "steps": int(steps),
                    "input_smiles": str(smiles),
                    "final_smiles": str(
                        md.extra.get("cli_compat", {}).get(
                            "final_smiles_raw", final_can
                        )
                    ),
                    "final_smiles_canonical": str(final_can),
                    "final_score": float(
                        md.extra.get("cli_compat", {}).get("final_score", 0.0)
                    ),
                    "actions": md.extra.get("cli_compat", {}).get("actions_raw", []),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    final_comp_merged: Optional[dict[str, float]]
    if final_components is None:
        final_comp_merged = {"rollout_reward": float(score_f)}
    else:
        final_comp_merged = dict(final_components)
        final_comp_merged["rollout_reward"] = float(score_f)

    return RunResult(
        lead=MoleculeRecord(
            smiles=lead_can,
            objective=float(lead_obj),
            components=lead_components,
            metadata={"role": "lead", "scoring": lead_score_md},
        ),
        final=MoleculeRecord(
            smiles=final_can or str(final_smiles),
            objective=float(final_obj_sc),
            components=final_comp_merged,
            metadata={"role": "final", "scoring": final_score_md},
        ),
        trace=ActionTrace(steps=trace_steps, terminated=True, length=len(trace_steps)),
        metadata=md,
    )
