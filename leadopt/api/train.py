"""PPO training API (stable importable). Mirrors `leadopt train`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .presets import resolve_preset_path
from .types import TrainResult


def train(
    *,
    preset: Optional[Union[str, Path]] = None,
    dataset: Optional[Union[str, Path]] = None,
    smiles: Optional[str] = None,
    smiles_col: str = "smiles",
    dataset_limit: int = 0,
    split_test_size: float = 0.2,
    split_random_state: int = 0,
    total_updates: int = 200,
    eval_every: int = 10,
    eval_episodes_per_lead: int = 8,
    save_every: int = 10,
    keep_last_k: int = 5,
    seed: Optional[int] = None,
    run_dir: Optional[Union[str, Path]] = None,
    write_artifacts: bool = True,
    verbose: bool = False,
    device: Optional[str] = None,
    resume_from: Optional[Union[str, Path]] = None,
) -> TrainResult:
    if run_dir is None:
        raise ValueError("train(...) requires run_dir (output directory) to be set.")

    if (dataset is None) == (smiles is None):
        raise ValueError("train(...) requires exactly one of: dataset or smiles.")

    preset_path = resolve_preset_path(preset)
    preset_token = str(preset) if preset is not None else ""

    from leadopt.engine.train_ppo import train_dataset, train_single

    _ = verbose

    if dataset is not None:
        return train_dataset(
            run_dir=Path(run_dir),
            preset=preset_token,
            seed=seed,
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
            resume_from=resume_from,
            device=device,
            preset_path_for_metadata=preset_path,
            write_artifacts=bool(write_artifacts),
        )

    assert smiles is not None
    return train_single(
        run_dir=Path(run_dir),
        preset=preset_token,
        seed=seed,
        smiles=smiles,
        total_updates=total_updates,
        eval_every=eval_every,
        eval_episodes_per_lead=eval_episodes_per_lead,
        save_every=save_every,
        keep_last_k=keep_last_k,
        resume_from=resume_from,
        device=device,
        preset_path_for_metadata=preset_path,
        write_artifacts=bool(write_artifacts),
    )


__all__ = ["train"]
