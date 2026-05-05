from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from leadopt.cli._preset_path import resolve_preset_path


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="leadopt-run",
        description="Run a leadopt preset on a single SMILES (smoke + teaching aid).",
    )
    ap.add_argument(
        "--preset",
        type=str,
        required=True,
        help=(
            "Preset YAML path. Accepts filesystem path or package resource, e.g. "
            "'leadopt/presets/medchem_quality_tier4.yaml'."
        ),
    )
    ap.add_argument("--smiles", type=str, required=True, help="Input SMILES.")
    ap.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Max environment steps for the random rollout (default: 3).",
    )
    ap.add_argument("--seed", type=int, default=0, help="Global seed (default: 0).")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output directory. Writes result.json if provided.",
    )
    ap.add_argument(
        "--rdkit-quiet",
        action="store_true",
        help="Disable RDKit logging (rdApp.*).",
    )
    return ap


def _cli_result_from_runresult(
    res: Any, *, input_smiles: str, steps: int
) -> Dict[str, Any]:
    """Preserve the historical CLI JSON output format."""
    extra = getattr(getattr(res, "metadata", None), "extra", {}) or {}
    cli_compat = extra.get("cli_compat", {}) if isinstance(extra, dict) else {}
    return {
        "seed": int(res.metadata.seed),
        "steps": int(steps),
        "input_smiles": str(input_smiles),
        "final_smiles": str(cli_compat.get("final_smiles_raw", res.final.smiles)),
        "final_smiles_canonical": str(res.final.smiles),
        "final_score": float(cli_compat.get("final_score", 0.0)),
        "actions": cli_compat.get("actions_raw", []),
    }


def main(argv: Optional[list[str]] = None) -> None:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)

    if args.rdkit_quiet:
        try:
            from rdkit import RDLogger

            RDLogger.DisableLog("rdApp.*")
        except ImportError:
            # Engine will raise a clearer error later if RDKit is required.
            pass

    with resolve_preset_path(args.preset) as preset_path:
        from leadopt.engine.run_rollout import run_once

        rr = run_once(
            preset_path=Path(preset_path),
            smiles=str(args.smiles),
            steps=int(args.steps),
            seed=int(args.seed),
            preset_name=str(args.preset),
        )

    res = _cli_result_from_runresult(
        rr, input_smiles=str(args.smiles), steps=int(args.steps)
    )

    print(json.dumps(res, indent=2, sort_keys=True))
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(
            json.dumps(res, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
