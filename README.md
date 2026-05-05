# leadopt

Academic molecular reinforcement learning package for lead optimization.

leadopt provides:

- YAML-driven experiment configuration
- Deterministic operator contracts
- Failure-safe scoring (QSAR, MPO, Docking)
- PPO-based policy training
- CLI and Python API workflows
- Reproducibility-first design (preset snapshot + run manifest)

---

# Installation

leadopt supports modular installation depending on your workflow.

---

## Minimal (No RDKit / No Torch)

Installs:

- Configuration scaffolding
- Packaged presets
- Static data files

Does **not** enable chemistry or RL execution.

```bash
pip install .
leadopt-sanity --data-only
```

Use this mode for:

- Inspecting presets
- Verifying installation
- CI checks without heavy dependencies

---

## Chemistry Backend (RDKit)

Required for:

- `leadopt run`
- `leadopt beam`
- SMILES rollouts
- Constraint evaluation
- Most scoring functions

```bash
pip install ".[chem]"
leadopt sanity
```

---

## RL Backend (Torch)

Required for:

- `leadopt train`
- `leadopt generate`

```bash
pip install ".[rl]"
```

---

## Full Installation (Recommended for Development)

```bash
pip install ".[all,dev]"
pytest -q
```

Installs:

- RDKit
- Torch
- Development tools
- Test dependencies

---

# CLI Quickstart

Use the umbrella CLI:

```
leadopt ...
```

Backwards-compatible entrypoints remain available:

- `leadopt-run`
- `leadopt-sanity`

---

## Verify Packaged Presets

```bash
python -c "import importlib.resources as r; print(sorted([p.name for p in r.files('leadopt').joinpath('presets').iterdir()]))"
```

---

# Run a Preset Rollout (RDKit Required)

```bash
leadopt run \
  --preset medchem_quality_tier4 \
  --smiles "CCOCC" \
  --steps 2 \
  --seed 0
```

This performs a single environment rollout and prints the final molecule and score.

---

# Beam Search Generation (RDKit Required)

```bash
leadopt beam \
  --preset np_fragment_discovery \
  --smiles "CC(=O)NCc1ccccc1" \
  --beam_width 10 \
  --max_steps 4 \
  --top_n 50 \
  --out_csv runs/beam.csv
```

Beam-specific knobs are configured in the preset YAML under:

```
beam:
```

See `docs/cli.md` for details.

---

# PPO Training (Torch + RDKit Required)

## Dataset Mode

```bash
leadopt train \
  --preset medchem_quality_tier4 \
  --dataset dataset.csv \
  --smiles_col smiles \
  --out_dir runs/ppo_run \
  --total_updates 200 \
  --save_every 10 \
  --keep_last_k 5
```

---

## Single-Lead Mode

```bash
leadopt train \
  --preset medchem_quality_tier4 \
  --smiles "c1ccc(cc1)O" \
  --out_dir runs/ppo_single \
  --total_updates 50
```

---

# Documentation

Detailed documentation is available in:

- `docs/cli.md`
- `docs/api.md`
- `docs/presets.md`
- `docs/operators.md`
- `docs/scorers.md`
- `docs/constraints.md`

---

# Reproducibility

Each training run directory should contain:

- `preset_used.yaml`
- `run_manifest.json`
- Checkpoints
- Operator signature
- Vocabulary (if applicable)

leadopt is designed so that:

- Presets fully define experiments
- Operator behavior is deterministic
- Scoring failures are explicit and logged
- Results are versioned and auditable

For academic use, always archive the full run directory.
