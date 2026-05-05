"""Run rollout API (stable importable).

This mirrors `leadopt run` but returns a stable RunResult dataclass.

Design constraints:
- CLI remains source-of-truth; API calls shared engine paths.
- Keep imports light at module import time (heavy deps imported inside functions).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .presets import resolve_preset_path
from .types import RunMetadata, RunResult


def run(
    *,
    preset: Optional[Union[str, Path]] = None,
    smiles: str,
    steps: int,
    seed: int = 0,
    run_dir: Optional[Union[str, Path]] = None,
    write_artifacts: bool = True,
    verbose: bool = False,
    device: Optional[str] = None,
) -> RunResult:
    """Run a single policy/environment rollout.

    Mirrors `leadopt run`.

    Args:
        preset: Preset name or YAML path.
        smiles: Starting molecule SMILES.
        steps: Max env steps for rollout.
        seed: Global seed.
        run_dir: Optional directory to associate with this run (Phase 2.4 will
            standardize artifact writing here).
        write_artifacts: Reserved for Phase 2.4 (currently no-op for run()).
        verbose: Reserved for Phase 2.4/2.3 CLI parity (currently no-op).
        device: Optional device string (recorded in metadata).

    Returns:
        RunResult
    """
    preset_path = resolve_preset_path(preset)
    if preset_path is None:
        raise ValueError(
            "run(...) requires a preset name/path (preset=None is not supported)."
        )

    from leadopt.engine.run_rollout import run_once

    rr = run_once(
        preset_path=Path(preset_path),
        smiles=str(smiles),
        steps=int(steps),
        seed=int(seed),
        device=device,
        preset_name=str(preset) if preset is not None else None,
        run_dir=Path(run_dir) if run_dir is not None else None,
        write_artifacts=bool(write_artifacts),
    )

    # Phase 2.4 will implement artifact writing. For now, just record run_dir if provided.
    if run_dir is not None:
        md = RunMetadata(
            seed=rr.metadata.seed,
            device=rr.metadata.device,
            preset_name=rr.metadata.preset_name,
            preset_path=rr.metadata.preset_path,
            run_dir=str(Path(run_dir)),
            extra=rr.metadata.extra,
        )
        rr = RunResult(lead=rr.lead, final=rr.final, trace=rr.trace, metadata=md)

    # write_artifacts and verbose currently do not change behavior for run()
    _ = write_artifacts
    _ = verbose

    return rr


__all__ = ["run"]
