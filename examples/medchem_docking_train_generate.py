# FILE: examples/medchem_docking_train_generate.py

from __future__ import annotations

from pathlib import Path

from leadopt.api import generate, train

# ---------------------------------------------------------------------
# Reproducible knobs
# ---------------------------------------------------------------------
SEED = 0

# IMPORTANT:
# Replace this with your docking preset name/path if different.
# Example: PRESET = "medchem_docking_vina" or "leadopt/presets/medchem_docking_vina.yaml"
PRESET = "medchem_quality_tier4"

LEAD_SMILES = "CCO"

# Keep tiny by default (smoke run). Increase for real docking optimization.
TOTAL_UPDATES = 1
EVAL_EVERY = 1
SAVE_EVERY = 1

EPISODES = 8
TOP_K = 10
POLICY = "sample"

RUN_DIR = Path("runs/examples/medchem_docking_train_generate")
WRITE_ARTIFACTS = False


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    tr = train(
        preset=PRESET,
        smiles=LEAD_SMILES,
        run_dir=RUN_DIR,
        seed=SEED,
        total_updates=TOTAL_UPDATES,
        eval_every=EVAL_EVERY,
        save_every=SAVE_EVERY,
        write_artifacts=WRITE_ARTIFACTS,
    )

    ckpt = tr.last_checkpoint or tr.best_checkpoint
    ckpt_token = Path(ckpt).name if ckpt else "model_last.pt"

    gr = generate(
        preset=PRESET,
        run_dir=RUN_DIR,
        checkpoint=ckpt_token,
        smiles=LEAD_SMILES,
        seed=SEED,
        episodes=EPISODES,
        top_k=TOP_K,
        policy=POLICY,
        write_artifacts=WRITE_ARTIFACTS,
    )

    print("\n=== Docking pipeline (template) ===")
    print("Preset:", PRESET)
    print("Train run_dir:", tr.run_dir)
    print("Checkpoint:", ckpt_token)
    print("\nTop candidates:")
    for i, rec in enumerate(gr.candidates[:10], 1):
        comps = rec.components or {}
        print(f"{i:02d}. obj={rec.objective:.4f} smiles={rec.smiles} comps={list(comps.keys())[:5]}")
    print("Generate artifacts:", gr.artifacts)

    print(
        "\nNOTE: If you intended a docking scorer and this preset is not docking-enabled, "
        "replace PRESET with your docking preset name/path."
    )


if __name__ == "__main__":
    main()