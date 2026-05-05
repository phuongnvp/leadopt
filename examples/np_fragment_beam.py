# FILE: examples/np_fragment_beam.py

from __future__ import annotations

from pathlib import Path

from leadopt.api import beam

# ---------------------------------------------------------------------
# Reproducible knobs
# ---------------------------------------------------------------------
SEED = 0
PRESET = "preset.yaml"
LEAD_SMILES = "C=C[C@H]1CN2CC[C@H]1C[C@@H]2[C@H](c3ccnc4c3cc(OC)cc4)O"

# Small beam by default
BEAM_WIDTH = 4
MAX_STEPS = 2
TOP_N = 20

RUN_DIR = Path("results")
WRITE_ARTIFACTS = True  # beam is fast; leaving True is useful for labs


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    br = beam(
        preset=PRESET,
        smiles=LEAD_SMILES,
        seed=SEED,
        beam_width=BEAM_WIDTH,
        max_steps=MAX_STEPS,
        top_n=TOP_N,
        run_dir=RUN_DIR,
        write_artifacts=WRITE_ARTIFACTS,
    )

    print("\n=== BeamResult ===")
    print("Preset:", PRESET)
    print("Lead:", br.lead.smiles)
    print("Num candidates:", len(br.candidates))
    print("Artifacts:", br.artifacts)

    print("\nTop candidates:")
    for i, rec in enumerate(br.candidates[:10], 1):
        comps = rec.components or {}
        aug = comps.get("augmented_objective", None)
        cx = comps.get("complexity", None)
        print(
            f"{i:02d}. obj={rec.objective:.4f} "
            f"aug={aug:.4f}" if isinstance(aug, float) else f"{i:02d}. obj={rec.objective:.4f} aug=?",
            f"cx={cx:.1f}" if isinstance(cx, float) else "cx=?",
            f"smiles={rec.smiles}",
        )


if __name__ == "__main__":
    main()