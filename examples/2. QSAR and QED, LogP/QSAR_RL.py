from __future__ import annotations

from pathlib import Path

from leadopt.api import generate, train

# ---------------------------------------------------------------------
# Reproducible knobs
# ---------------------------------------------------------------------
SEED = 0
PRESET = "medchem_quality_tier4"
LEAD_SMILES = "CCO"
TOTAL_UPDATES = 1 # 500
EVAL_EVERY = 1 # 25
SAVE_EVERY = 1 # 50
EPISODES = 16 # 128
TOP_K = 10 # 100
POLICY = "sample"
RUN_DIR = Path("runs/examples/medchem_qsar_train_generate")
WRITE_ARTIFACTS = True


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

    print("\n=== TrainResult ===")
    print("run_dir:", tr.run_dir)
    print("best_checkpoint:", tr.best_checkpoint)
    print("last_checkpoint:", tr.last_checkpoint)
    print("num_checkpoints:", len(tr.checkpoints))
    print("artifacts:", tr.artifacts)

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

    print("\n=== GenerateResult ===")
    print("unique_count:", gr.unique_count)
    print("top candidates:")
    for i, rec in enumerate(gr.candidates[:10], 1):
        print(f"{i:02d}. obj={rec.objective:.4f} smiles={rec.smiles}")
    print("artifacts:", gr.artifacts)


if __name__ == "__main__":
    main()