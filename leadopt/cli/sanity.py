from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path
from typing import Optional

from leadopt.cli._preset_path import resolve_preset_path
from leadopt.core.seeding import set_global_seed


def _has_rdkit() -> bool:
    try:
        import rdkit  # noqa: F401

        return True
    except ImportError:
        return False


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="leadopt-sanity",
        description="Minimal install + data self-check for leadopt.",
    )
    ap.add_argument(
        "--preset",
        type=str,
        default="leadopt/presets/medchem_quality_tier4.yaml",
        help="Preset to load for sanity (default: medchem_quality_tier4).",
    )
    ap.add_argument(
        "--smiles",
        type=str,
        default="CCOCC",
        help="SMILES for sanity rollout (default: CCOCC).",
    )
    ap.add_argument("--steps", type=int, default=2, help="Rollout steps (default: 2).")
    ap.add_argument("--seed", type=int, default=0, help="Seed (default: 0).")

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--data-only",
        action="store_true",
        help="Only verify installed package data (no RDKit required).",
    )
    mode.add_argument(
        "--rollout",
        action="store_true",
        help="Force a tiny rollout (requires RDKit).",
    )
    return ap


def main(argv: Optional[list[str]] = None) -> None:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)

    # 1) Verify presets are discoverable as installed package data
    presets_dir = files("leadopt").joinpath("presets")
    if not presets_dir.is_dir():
        raise RuntimeError("Package data missing: leadopt/presets directory not found.")
    preset_files = sorted(
        [p.name for p in presets_dir.iterdir() if p.name.endswith(".yaml")]
    )
    if len(preset_files) == 0:
        raise RuntimeError(
            "Package data missing: no *.yaml files under leadopt/presets."
        )

    # Decide whether to run a rollout:
    # - If user requests --data-only: never roll out.
    # - If user requests --rollout: require RDKit.
    # - Default: roll out iff RDKit is available.
    do_rollout = False
    if args.data_only:
        do_rollout = False
    elif args.rollout:
        if not _has_rdkit():
            raise RuntimeError(
                "Rollout requested but RDKit is not installed. Install with "
                "'pip install leadopt[chem]' or via conda-forge "
                "('conda install -c conda-forge rdkit')."
            )
        do_rollout = True
    else:
        do_rollout = _has_rdkit()

    if do_rollout:
        # 2) Load a preset and build env (this path requires RDKit because PresetLoader
        # imports operator modules which depend on RDKit).
        with resolve_preset_path(args.preset) as preset_path:
            from leadopt.config.preset_loader import PresetLoader

            loader = PresetLoader()
            loaded = loader.load(Path(preset_path))

            set_global_seed(int(args.seed), deterministic_torch=True)

            env = loader.build_env(loaded)
            env.reset(str(args.smiles))
            _ = env.rollout_random(max_env_steps=int(args.steps))

    print("leadopt sanity OK")
    print(f"  presets_found: {len(preset_files)}")
    if do_rollout:
        print(f"  preset_loaded: {args.preset}")
        print(f"  rollout_steps: {args.steps}")
    else:
        print("  rollout: skipped (RDKit not installed or --data-only)")


if __name__ == "__main__":
    main()
