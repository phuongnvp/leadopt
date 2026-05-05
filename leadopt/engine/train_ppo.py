from __future__ import annotations

"""Shared PPO training engine for `leadopt train` and `leadopt.api.train`.

Design constraints:
- No algorithm changes.
- Reuse the existing, battle-tested training implementation from leadopt.cli.train
  while presenting stable API return objects.

Why this file exists:
- The CLI remains the source-of-truth for behavior.
- The engine becomes the shared execution path for both CLI and API.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from leadopt.api.types import RunMetadata, TrainResult


def _collect_train_artifacts(run_dir: Path) -> Dict[str, str]:
    artifacts: Dict[str, str] = {}
    for name in ["preset_used.yaml", "run_manifest.json", "run_config.json"]:
        p = run_dir / name
        if p.exists():
            artifacts[name] = str(p)
    sar_dir = run_dir / "sar_report"
    if sar_dir.exists() and sar_dir.is_dir():
        artifacts["sar_report_dir"] = str(sar_dir)
    return artifacts


def _prune_optional_artifacts(run_dir: Path) -> None:
    """Remove optional run-logging artifacts while keeping essential outputs.

    Policy:
    - Keep: checkpoints, vocab.json, any model weights, optimizer states, etc.
    - Remove (optional logging artifacts):
        - preset_used.yaml
        - run_manifest.json
        - run_config.json
        - sar_report/ directory

    This is used to implement API mode `write_artifacts=False` without refactoring
    the underlying training implementation.
    """
    for name in ["preset_used.yaml", "run_manifest.json", "run_config.json"]:
        p = run_dir / name
        try:
            if p.exists():
                p.unlink()
        except Exception:
            # Best-effort cleanup; do not fail training result.
            pass

    sar_dir = run_dir / "sar_report"
    try:
        if sar_dir.exists() and sar_dir.is_dir():
            # Python 3.8+: use shutil.rmtree
            import shutil

            shutil.rmtree(sar_dir, ignore_errors=True)
    except Exception:
        pass


def _collect_checkpoints(run_dir: Path) -> Dict[str, Any]:
    best = run_dir / "model_best.pt"
    last = run_dir / "model_last.pt"

    ckpts: List[str] = []

    # periodic checkpoints (CLI uses checkpoint_update_*.pt; keep legacy model_*.pt too)
    for p in sorted(run_dir.glob("checkpoint_update_*.pt")):
        ckpts.append(str(p))
    for p in sorted(run_dir.glob("model_*.pt")):
        if str(p) not in ckpts:
            ckpts.append(str(p))

    # Ensure best/last included in list (stable ordering)
    if best.exists() and str(best) not in ckpts:
        ckpts.insert(0, str(best))
    if last.exists() and str(last) not in ckpts:
        ckpts.append(str(last))

    return {
        "best": str(best) if best.exists() else None,
        "last": str(last) if last.exists() else None,
        "all": ckpts,
    }


def _namespace_for_train(
    *,
    out_dir: Union[str, Path],
    preset: str,
    seed: Optional[int],
    smiles: Optional[str],
    dataset: Optional[Union[str, Path]],
    smiles_col: str,
    dataset_limit: int,
    split_test_size: float,
    split_random_state: int,
    total_updates: int,
    eval_every: int,
    eval_episodes_per_lead: int,
    save_every: int,
    keep_last_k: int,
    resume: Union[str, Path, None],
) -> argparse.Namespace:
    # Matches leadopt.cli.train argument names exactly.
    return argparse.Namespace(
        out_dir=str(out_dir),
        run_dir=None,  # legacy alias unused when out_dir provided
        preset=str(preset),
        seed=seed,
        smiles=str(smiles) if smiles is not None else None,
        dataset=str(dataset) if dataset is not None else None,
        smiles_col=str(smiles_col),
        dataset_limit=int(dataset_limit),
        split_test_size=float(split_test_size),
        split_random_state=int(split_random_state),
        total_updates=int(total_updates),
        eval_every=int(eval_every),
        eval_episodes_per_lead=int(eval_episodes_per_lead),
        save_every=int(save_every),
        keep_last_k=int(keep_last_k),
        resume=str(resume) if resume is not None else "",
    )


def train_dataset(
    *,
    run_dir: Union[str, Path],
    preset: str = "",
    seed: Optional[int] = None,
    dataset: Union[str, Path],
    smiles_col: str = "smiles",
    dataset_limit: int = 0,
    split_test_size: float = 0.2,
    split_random_state: int = 0,
    total_updates: int = 200,
    eval_every: int = 10,
    eval_episodes_per_lead: int = 8,
    save_every: int = 10,
    keep_last_k: int = 5,
    resume_from: Union[str, Path, None] = None,
    device: Optional[str] = None,
    preset_path_for_metadata: Optional[Union[str, Path]] = None,
    write_artifacts: bool = True,
) -> TrainResult:
    """Train PPO on a dataset of leads (CSV). Mirrors CLI dataset mode."""
    from leadopt.cli.train import (  # type: ignore
        _require_optional_deps,
        _train_dataset_mode,
    )

    run_dir_p = Path(run_dir)
    run_dir_p.mkdir(parents=True, exist_ok=True)

    args = _namespace_for_train(
        out_dir=run_dir_p,
        preset=str(preset),
        seed=seed,
        smiles=None,
        dataset=dataset,
        smiles_col=smiles_col,
        dataset_limit=dataset_limit,
        split_test_size=split_test_size,
        split_random_state=split_random_state,
        total_updates=total_updates,
        eval_every=eval_every,
        eval_episodes_per_lead=eval_episodes_per_lead,
        save_every=save_every,
        keep_last_k=keep_last_k,
        resume=resume_from,
    )

    _require_optional_deps(dataset_mode=True)
    _train_dataset_mode(args, out_dir=run_dir_p)

    if not bool(write_artifacts):
        _prune_optional_artifacts(run_dir_p)

    ck = _collect_checkpoints(run_dir_p)
    artifacts = _collect_train_artifacts(run_dir_p) if bool(write_artifacts) else {}

    md = RunMetadata(
        seed=int(seed) if seed is not None else 0,
        device=device,
        preset_name=str(preset) if str(preset).strip() else None,
        preset_path=(
            str(preset_path_for_metadata)
            if preset_path_for_metadata is not None
            else None
        ),
        run_dir=str(run_dir_p),
        extra={
            "mode": "dataset",
            "dataset": {
                "path": str(dataset),
                "smiles_col": str(smiles_col),
                "dataset_limit": int(dataset_limit),
                "split_test_size": float(split_test_size),
                "split_random_state": int(split_random_state),
            },
        },
    )

    train_summary: Dict[str, float] = {
        "total_updates": float(total_updates),
        "eval_every": float(eval_every),
        "eval_episodes_per_lead": float(eval_episodes_per_lead),
    }

    return TrainResult(
        run_dir=str(run_dir_p),
        best_checkpoint=ck["best"],
        last_checkpoint=ck["last"],
        checkpoints=list(ck["all"]),
        train_summary=train_summary,
        metadata=md,
        artifacts=artifacts,
    )


def train_single(
    *,
    run_dir: Union[str, Path],
    preset: str = "",
    seed: Optional[int] = None,
    smiles: str,
    total_updates: int = 200,
    eval_every: int = 10,
    eval_episodes_per_lead: int = 8,
    save_every: int = 10,
    keep_last_k: int = 5,
    resume_from: Union[str, Path, None] = None,
    device: Optional[str] = None,
    preset_path_for_metadata: Optional[Union[str, Path]] = None,
    write_artifacts: bool = True,
) -> TrainResult:
    """Train PPO on a single lead. Mirrors CLI single mode."""
    from leadopt.cli.train import (  # type: ignore
        _require_optional_deps,
        _train_single_mode,
    )

    run_dir_p = Path(run_dir)
    run_dir_p.mkdir(parents=True, exist_ok=True)

    args = _namespace_for_train(
        out_dir=run_dir_p,
        preset=str(preset),
        seed=seed,
        smiles=smiles,
        dataset=None,
        smiles_col="smiles",
        dataset_limit=0,
        split_test_size=0.2,
        split_random_state=0,
        total_updates=total_updates,
        eval_every=eval_every,
        eval_episodes_per_lead=eval_episodes_per_lead,
        save_every=save_every,
        keep_last_k=keep_last_k,
        resume=resume_from,
    )

    _require_optional_deps(dataset_mode=False)
    _train_single_mode(args, out_dir=run_dir_p)

    if not bool(write_artifacts):
        _prune_optional_artifacts(run_dir_p)

    ck = _collect_checkpoints(run_dir_p)
    artifacts = _collect_train_artifacts(run_dir_p) if bool(write_artifacts) else {}

    md = RunMetadata(
        seed=int(seed) if seed is not None else 0,
        device=device,
        preset_name=str(preset) if str(preset).strip() else None,
        preset_path=(
            str(preset_path_for_metadata)
            if preset_path_for_metadata is not None
            else None
        ),
        run_dir=str(run_dir_p),
        extra={"mode": "single", "lead_smiles": str(smiles)},
    )

    train_summary: Dict[str, float] = {
        "total_updates": float(total_updates),
        "eval_every": float(eval_every),
        "eval_episodes_per_lead": float(eval_episodes_per_lead),
    }

    return TrainResult(
        run_dir=str(run_dir_p),
        best_checkpoint=ck["best"],
        last_checkpoint=ck["last"],
        checkpoints=list(ck["all"]),
        train_summary=train_summary,
        metadata=md,
        artifacts=artifacts,
    )
