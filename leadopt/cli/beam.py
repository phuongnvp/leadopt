from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from leadopt.cli._preset_path import resolve_preset_path
from leadopt.engine.beam_search import beam_search


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="leadopt beam",
        description=(
            "Deterministic beam search generation using a leadopt YAML preset (operators + scorer). "
            "Beam-specific parameters (complexity_weight, dock_drop_tolerance, hard_constraint_filter) "
            "are read from the preset under the 'beam:' section."
        ),
    )
    ap.add_argument(
        "--preset",
        type=str,
        required=True,
        help=(
            "Preset YAML path or shipped preset name. "
            "Examples: 'np_fragment_discovery' or 'leadopt/presets/np_fragment_discovery.yaml'."
        ),
    )
    ap.add_argument(
        "--smiles", type=str, default=None, help="Starting molecule SMILES."
    )
    ap.add_argument(
        "--beam_width",
        type=int,
        default=None,
        help="Beam width (default: preset beam.beam_width, else 20).",
    )
    ap.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Search depth (default: preset beam.max_steps, else 8).",
    )
    ap.add_argument(
        "--per_state_action_limit",
        type=int,
        default=None,
        help="Max expansions per beam state (default: preset beam.per_state_action_limit, else 256).",
    )
    ap.add_argument("--seed", type=int, default=0, help="Seed (default: 0).")
    ap.add_argument(
        "--out_csv",
        type=str,
        default="runs/beam_results.csv",
        help="Output CSV path (default: runs/beam_results.csv).",
    )
    ap.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="How many top results to write (default: preset beam.top_n, else 50).",
    )

    # Deprecated legacy flags (hidden)
    ap.add_argument(
        "--start-smiles",
        dest="smiles_legacy",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--beam-width",
        dest="beam_width_legacy",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--max-steps",
        dest="max_steps_legacy",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--per-state-action-limit",
        dest="per_state_action_limit_legacy",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--complexity-weight",
        dest="complexity_weight_legacy",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--dock-drop-tolerance",
        dest="dock_drop_tolerance_legacy",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--no-hard-constraint-filter",
        dest="no_hard_constraint_filter_legacy",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--out-csv",
        dest="out_csv_legacy",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--top-n", dest="top_n_legacy", type=int, default=None, help=argparse.SUPPRESS
    )

    return ap


def _beam_cfg_from_loaded(loaded: Any) -> Dict[str, Any]:
    tm = getattr(loaded, "training_meta", {})
    beam = tm.get("beam", {}) if isinstance(tm, dict) else {}
    if not isinstance(beam, dict):
        beam = {}
    return dict(beam)


def main(argv: Optional[list[str]] = None) -> None:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)

    # Legacy aliases for stable CLI flags.
    if args.smiles_legacy is not None:
        args.smiles = str(args.smiles_legacy)
        warnings.warn(
            "--start-smiles is deprecated; use --smiles instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if not args.smiles:
        raise SystemExit("Missing required argument: --smiles")

    if args.beam_width_legacy is not None:
        args.beam_width = int(args.beam_width_legacy)
        warnings.warn(
            "--beam-width is deprecated; use --beam_width.",
            DeprecationWarning,
            stacklevel=2,
        )
    if args.max_steps_legacy is not None:
        args.max_steps = int(args.max_steps_legacy)
        warnings.warn(
            "--max-steps is deprecated; use --max_steps.",
            DeprecationWarning,
            stacklevel=2,
        )
    if args.per_state_action_limit_legacy is not None:
        args.per_state_action_limit = int(args.per_state_action_limit_legacy)
        warnings.warn(
            "--per-state-action-limit is deprecated; use --per_state_action_limit.",
            DeprecationWarning,
            stacklevel=2,
        )
    if args.out_csv_legacy is not None:
        args.out_csv = str(args.out_csv_legacy)
        warnings.warn(
            "--out-csv is deprecated; use --out_csv.", DeprecationWarning, stacklevel=2
        )
    if args.top_n_legacy is not None:
        args.top_n = int(args.top_n_legacy)
        warnings.warn(
            "--top-n is deprecated; use --top_n.", DeprecationWarning, stacklevel=2
        )

    with resolve_preset_path(args.preset) as preset_path:
        # Pull YAML-driven defaults
        from leadopt.config.preset_loader import PresetLoader

        loaded = PresetLoader().load(Path(preset_path))
        beam_cfg = _beam_cfg_from_loaded(loaded)

        complexity_weight = float(beam_cfg.get("complexity_weight", 0.05))
        dock_drop_tolerance = beam_cfg.get("dock_drop_tolerance", None)
        if dock_drop_tolerance is not None:
            dock_drop_tolerance = float(dock_drop_tolerance)
        hard_constraint_filter = bool(beam_cfg.get("hard_constraint_filter", True))

        # Legacy overrides
        if args.complexity_weight_legacy is not None:
            complexity_weight = float(args.complexity_weight_legacy)
            warnings.warn(
                "--complexity-weight is deprecated; configure in preset 'beam:' section.",
                DeprecationWarning,
                stacklevel=2,
            )
        if args.dock_drop_tolerance_legacy is not None:
            dock_drop_tolerance = float(args.dock_drop_tolerance_legacy)
            warnings.warn(
                "--dock-drop-tolerance is deprecated; configure in preset 'beam:' section.",
                DeprecationWarning,
                stacklevel=2,
            )
        if args.no_hard_constraint_filter_legacy:
            hard_constraint_filter = False
            warnings.warn(
                "--no-hard-constraint-filter is deprecated; configure in preset 'beam:' section.",
                DeprecationWarning,
                stacklevel=2,
            )

        out_path = Path(args.out_csv)
        br = beam_search(
            preset_path=Path(preset_path),
            smiles=str(args.smiles),
            seed=int(args.seed),
            beam_width=args.beam_width,
            max_steps=args.max_steps,
            per_state_action_limit=args.per_state_action_limit,
            top_n=args.top_n,
            out_csv=out_path,
            write_files=True,
            complexity_weight=complexity_weight,
            dock_drop_tolerance=dock_drop_tolerance,
            hard_constraint_filter=hard_constraint_filter,
            preset_name=str(args.preset),
        )

    pool_path = br.artifacts.get(
        "pool_jsonl", str(Path(args.out_csv).with_suffix(".pool.jsonl"))
    )
    print(f"Wrote init pool JSONL to {pool_path}")
    print(
        f"Wrote {len(br.candidates)} results to {br.artifacts.get('csv', str(args.out_csv))}"
    )

    if br.candidates:
        best = br.candidates[0]
        aug = best.components.get("augmented_objective") if best.components else None
        cx = best.components.get("complexity") if best.components else None
        print(
            "Best:",
            best.smiles,
            f"aug={aug:.3f}" if isinstance(aug, float) else "aug=?",
            f"obj={best.objective:.3f}",
            f"cx={cx:.1f}" if isinstance(cx, float) else "cx=?",
            f"step={best.metadata.get('step', '?')}",
        )


if __name__ == "__main__":
    main()
