"""Beam search API (stable importable).

This mirrors `leadopt beam` but returns a stable BeamResult dataclass.

Preset-driven defaults (CLI semantics):
- beam_width/max_steps/per_state_action_limit/top_n default to values in preset under `beam:`
- if not present in preset, fall back to CLI defaults (20/8/256/50)
- explicit API arguments override preset
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .presets import resolve_preset_path
from .types import BeamResult, RunMetadata


def beam(
    *,
    preset: Optional[Union[str, Path]] = None,
    smiles: str,
    seed: int = 0,
    # Phase 2.4 IO policy
    run_dir: Optional[Union[str, Path]] = None,
    write_artifacts: bool = True,
    verbose: bool = False,
    device: Optional[str] = None,
    # Beam params:
    # If None, they are loaded from preset `beam:` section (CLI semantics).
    beam_width: Optional[int] = None,
    max_steps: Optional[int] = None,
    per_state_action_limit: Optional[int] = None,
    top_n: Optional[int] = None,
) -> BeamResult:
    """Run beam search from a lead molecule.

    Args:
        preset: Preset name or YAML path.
        smiles: Starting molecule SMILES.
        seed: Seed.
        run_dir: Optional directory for artifacts. If provided and write_artifacts=True,
            writes `beam_results.csv` and `beam_results.pool.jsonl` into run_dir.
        write_artifacts: If True and run_dir is provided, write artifacts.
        verbose: Reserved; currently no-op (engine does not print).
        device: Optional device string (recorded in metadata).
        beam_width: If None, loaded from preset `beam.beam_width` else CLI default 20.
        max_steps: If None, loaded from preset `beam.max_steps` else CLI default 8.
        per_state_action_limit: If None, loaded from preset `beam.per_state_action_limit` else 256.
        top_n: If None, loaded from preset `beam.top_n` else 50.

    Returns:
        BeamResult
    """
    preset_path = resolve_preset_path(preset)
    if preset_path is None:
        raise ValueError(
            "beam(...) requires a preset name/path (preset=None is not supported)."
        )

    from leadopt.engine.beam_search import beam_search

    write_files = bool(run_dir is not None and write_artifacts)

    br = beam_search(
        preset_path=Path(preset_path),
        smiles=str(smiles),
        seed=int(seed),
        beam_width=beam_width,
        max_steps=max_steps,
        per_state_action_limit=per_state_action_limit,
        top_n=top_n,
        out_csv=None,  # engine will choose run_dir/beam_results.csv if run_dir is set
        run_dir=Path(run_dir) if run_dir is not None else None,
        write_files=write_files,
        device=device,
        preset_name=str(preset) if preset is not None else None,
    )

    if run_dir is not None:
        md = RunMetadata(
            seed=br.metadata.seed,
            device=br.metadata.device,
            preset_name=br.metadata.preset_name,
            preset_path=br.metadata.preset_path,
            run_dir=str(Path(run_dir)),
            extra=br.metadata.extra,
        )
        br = BeamResult(
            lead=br.lead, candidates=br.candidates, metadata=md, artifacts=br.artifacts
        )

    _ = verbose  # reserved for future parity hooks

    return br


__all__ = ["beam"]
