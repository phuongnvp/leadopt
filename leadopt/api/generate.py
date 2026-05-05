"""Policy rollout generation API (stable importable). Mirrors `leadopt generate`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .presets import resolve_preset_path
from .types import GenerateResult


def generate(
    *,
    preset: Optional[Union[str, Path]] = None,
    checkpoint: Union[str, Path],
    smiles: str,
    seed: int = 0,
    run_dir: Optional[Union[str, Path]] = None,
    write_artifacts: bool = True,
    verbose: bool = False,
    device: Optional[str] = None,
    episodes: int = 128,
    top_k: int = 50,
    policy: str = "sample",
    out_csv: Optional[Union[str, Path]] = None,
) -> GenerateResult:
    if run_dir is None:
        # For generate, run_dir is needed to find vocab.json and resolve relative ckpt paths.
        raise ValueError("generate(...) requires run_dir (training output directory).")

    preset_path = resolve_preset_path(preset)
    preset_token = str(preset) if preset is not None else ""

    from leadopt.engine.generate_ppo import generate_from_checkpoint

    _ = verbose

    return generate_from_checkpoint(
        run_dir=Path(run_dir),
        checkpoint=checkpoint,
        preset=preset_token,
        preset_path_for_metadata=preset_path,
        smiles=smiles,
        seed=int(seed),
        episodes=int(episodes),
        top_k=int(top_k),
        policy=str(policy),
        out_csv=out_csv,
        write_files=bool(write_artifacts),
        device=device,
    )


__all__ = ["generate"]
