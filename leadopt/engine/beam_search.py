from __future__ import annotations

"""Shared implementation for `leadopt beam` and `leadopt.api.beam`.

This engine mirrors CLI behavior:
- reads beam config from preset `beam:` section
- runs deterministic beam search
- optionally writes CSV + JSONL pool
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

from leadopt.actions.space import ActionSpace
from leadopt.api.types import BeamResult, MoleculeRecord, RunMetadata
from leadopt.config.preset_loader import PresetLoader
from leadopt.core.seeding import set_global_seed
from leadopt.evaluation.beam_decomplexify import beam_decomplexify


def _beam_cfg_from_loaded(loaded: Any) -> Dict[str, Any]:
    tm = getattr(loaded, "training_meta", {})
    beam = tm.get("beam", {}) if isinstance(tm, dict) else {}
    if not isinstance(beam, dict):
        beam = {}
    return dict(beam)


def beam_search(
    *,
    preset_path: Path,
    smiles: str,
    seed: int,
    beam_width: Optional[int] = None,
    max_steps: Optional[int] = None,
    per_state_action_limit: Optional[int] = None,
    top_n: Optional[int] = None,
    out_csv: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    write_files: bool = True,
    # Optional explicit overrides (CLI uses these for deprecated flags)
    complexity_weight: Optional[float] = None,
    dock_drop_tolerance: Optional[float] = None,
    hard_constraint_filter: Optional[bool] = None,
    device: Optional[str] = None,
    preset_name: Optional[str] = None,
) -> BeamResult:
    """Run beam search with a loaded preset."""

    set_global_seed(int(seed))

    loaded = PresetLoader().load(Path(preset_path))
    beam_cfg = _beam_cfg_from_loaded(loaded)

    # Defaults match historical CLI defaults.
    # Complexity weight resolution:
    # - Explicit CLI/API override wins
    # - Else use preset value if present
    # - Else default to 0.0 (no complexity penalty)

    if complexity_weight is not None:
        cx_w = float(complexity_weight)
    else:
        cx_w = float(beam_cfg.get("complexity_weight", 0.0))

    ddt = beam_cfg.get("dock_drop_tolerance", None)
    if ddt is not None:
        ddt = float(ddt)

    hcf = bool(beam_cfg.get("hard_constraint_filter", True))
    if dock_drop_tolerance is not None:
        ddt = float(dock_drop_tolerance)
    if hard_constraint_filter is not None:
        hcf = bool(hard_constraint_filter)

    gating_constraint = (
        loaded.legality_constraint_factory()
        if loaded.legality_constraint_factory
        else None
    )
    rule_config = loaded.env_kwargs.get("rule_config", None)
    action_space = ActionSpace(
        operators=loaded.operators,
        constraint=gating_constraint,
        rule_config=rule_config,
    )

    context: Optional[Dict[str, Any]] = {"seed": int(seed)}
    # Preset-driven defaults for search-shape parameters (CLI parity):
    bw = int(beam_cfg.get("beam_width", 20)) if beam_width is None else int(beam_width)
    ms = int(beam_cfg.get("max_steps", 8)) if max_steps is None else int(max_steps)
    pal = (
        int(beam_cfg.get("per_state_action_limit", 256))
        if per_state_action_limit is None
        else int(per_state_action_limit)
    )
    tn = int(beam_cfg.get("top_n", 50)) if top_n is None else int(top_n)

    items = beam_decomplexify(
        start_smiles=str(smiles),
        action_space=action_space,
        scorer=loaded.scorer,
        constraint_suite=loaded.constraint_suite,
        beam_width=bw,
        max_steps=ms,
        per_state_action_limit=pal,
        complexity_weight=float(cx_w),
        docking_drop_tolerance=ddt,
        hard_constraint_filter=bool(hcf),
        context=context,
    )
    rows = items[:tn]
    if out_csv is None and run_dir is not None and bool(write_files):
        out_csv = Path(run_dir) / "beam_results.csv"

    artifacts: Dict[str, str] = {}
    if out_csv is not None and bool(write_files):
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "rank",
                    "smiles",
                    "augmented_objective",
                    "objective",
                    "complexity",
                    "step",
                    "parent_smiles",
                    "action_operator",
                    "action_template",
                ]
            )
            for i, it in enumerate(rows, start=1):
                w.writerow(
                    [
                        i,
                        it.smiles,
                        f"{it.augmented_objective:.6f}",
                        f"{it.objective:.6f}",
                        f"{it.complexity:.6f}",
                        it.step,
                        it.parent_smiles or "",
                        it.action_operator or "",
                        it.action_template or "",
                    ]
                )

        pool_path = out_path.with_suffix(".pool.jsonl")
        with pool_path.open("w", encoding="utf-8") as f:
            for it in rows:
                f.write(
                    (
                        '{"smiles": %r, "objective": %.6f, "complexity": %.6f, "step": %d, '
                        '"parent_smiles": %r, "action_operator": %r, "action_template": %r}\n'
                    )
                    % (
                        it.smiles,
                        float(it.objective),
                        float(it.complexity),
                        int(it.step),
                        it.parent_smiles or "",
                        it.action_operator or "",
                        it.action_template or "",
                    )
                )

        artifacts["csv"] = str(out_path)
        artifacts["pool_jsonl"] = str(pool_path)

    candidates = []
    for it in rows:
        scoring_md = {
            "constraints": {
                str(k): float(v) for k, v in (it.constraints or {}).items()
            },
            "metadata": json.loads(json.dumps(it.metadata or {}, default=str)),
        }
        candidates.append(
            MoleculeRecord(
                smiles=str(it.smiles),
                objective=float(it.objective),
                components={
                    "augmented_objective": float(it.augmented_objective),
                    "complexity": float(it.complexity),
                },
                metadata={
                    "step": int(it.step),
                    "parent_smiles": it.parent_smiles or "",
                    "action_operator": it.action_operator or "",
                    "action_template": it.action_template or "",
                    "scoring": scoring_md,
                },
            )
        )

    md = RunMetadata(
        seed=int(seed),
        device=device,
        preset_name=preset_name,
        preset_path=str(preset_path),
        run_dir=None,
        extra={
            "beam": {
                "beam_width": int(bw),
                "max_steps": int(ms),
                "per_state_action_limit": int(pal),
                "top_n": int(tn),
                "complexity_weight": float(cx_w),
                "dock_drop_tolerance": ddt,
                "hard_constraint_filter": bool(hcf),
            }
        },
    )

    return BeamResult(
        lead=MoleculeRecord(
            smiles=str(smiles), objective=0.0, metadata={"role": "lead"}
        ),
        candidates=candidates,
        metadata=md,
        artifacts=artifacts,
    )
