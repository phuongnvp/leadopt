from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional
from leadopt.core.signatures import operator_signature

def _operator_signature(operators: List[Any]) -> str:
    return operator_signature(operators)

def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="leadopt generate",
        description=(
            "Generate molecules from a trained PPO policy.\n"
            "- Legacy mode (default): uses the built-in QED preset.\n"
            "- Preset mode (--preset): builds env/scorer from a YAML preset (QSAR/docking/etc.)."
        ),
    )
    ap.add_argument(
        "--run_dir", type=str, required=True, help="Training run directory."
    )

    ap.add_argument(
        "--preset",
        type=str,
        default="",
        help="Optional: YAML preset path or shipped preset name. If set, env/scorer comes from the preset.",
    )

    ap.add_argument(
        "--checkpoint",
        type=str,
        default="model_best",
        choices=["model_best", "model_last"],
        help="Checkpoint selector (default: model_best).",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=(
            "(alias) Checkpoint filename or path. "
            "Accepts: model_best.pt, model_last.pt, checkpoint_update_XXXXX.pt, or a full path. "
            "Overrides --checkpoint if provided."
        ),
    )

    ap.add_argument("--smiles", type=str, default=None, help="Seed SMILES.")
    ap.add_argument("--lead", type=str, default=None, help="(alias) Seed SMILES.")

    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument(
        "--policy", type=str, default="sample", choices=["greedy", "sample"]
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional output CSV path; defaults to run_dir naming.",
    )
    return ap


def main(argv: Optional[list[str]] = None) -> None:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)

    # Resolve smiles alias
    if args.smiles is None and args.lead is not None:
        args.smiles = str(args.lead)
    if not args.smiles:
        raise SystemExit("Missing required seed SMILES: --smiles (or legacy --lead)")

    run_dir = Path(str(args.run_dir))
    if not run_dir.exists():
        raise SystemExit(f"run_dir does not exist: {run_dir}")

    # Resolve checkpoint token (engine accepts both)
    ckpt = (
        args.ckpt
        if args.ckpt is not None
        else ("model_best.pt" if args.checkpoint == "model_best" else "model_last.pt")
    )

    from leadopt.engine.generate_ppo import generate_from_checkpoint

    res = generate_from_checkpoint(
        run_dir=run_dir,
        checkpoint=ckpt,
        checkpoint_name=str(args.checkpoint),
        preset=str(args.preset),
        smiles=str(args.smiles),
        seed=int(args.seed),
        episodes=int(args.episodes),
        top_k=int(args.top_k),
        policy=str(args.policy),
        out_csv=str(args.out_csv) if args.out_csv else None,
        write_files=True,
    )

    # Preserve CLI prints (summary + wrote path)
    lead = res.lead.smiles
    print(f"\nLead: {lead}")
    print(
        f"Unique molecules found: {int(res.unique_count)} / {int(args.episodes)} episodes"
    )
    print("\nTop candidates:")
    for i, rec in enumerate(res.candidates[:10], 1):
        print(f"{i:02d}. score={rec.objective:.4f}  smiles={rec.smiles}")

    if "csv" in res.artifacts:
        print(f"\nWrote: {res.artifacts['csv']}")
