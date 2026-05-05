# leadopt CLI

> Recommended: use `leadopt ...`  
> Legacy entrypoints remain available:
>
> - `leadopt-run` (equivalent to `leadopt run`)
> - `leadopt-sanity` (equivalent to `leadopt sanity`)

---

## Global Notes

### Preset Resolution

`--preset` accepts either:

- A file path  
  `--preset /abs/path/to/custom.yaml`
- A shipped preset name  
  `--preset medchem_quality_tier4`

Shipped presets live under:

```
leadopt/presets/*.yaml
```

---

## `leadopt sanity`

Verifies that packaged presets and data files are accessible.  
If RDKit is installed, it can also run a minimal rollout.

```bash
leadopt sanity --data-only
leadopt sanity --rollout --smiles "CCO" --steps 1
```

---

## `leadopt run`

Runs a single rollout from a starting SMILES using a preset-defined environment and scorer.

```bash
leadopt run \
  --preset medchem_quality_tier4 \
  --smiles "CCOCC" \
  --steps 2 \
  --seed 0
```

Outputs include:

- Final molecule
- Objective score
- Optional JSON artifacts (depending on flags and engine behavior)

---

## `leadopt train`

Trains a PPO policy for lead optimization.

### Input Modes (Mutually Exclusive)

### 1. Dataset Mode

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

Optional dataset controls:

- `--dataset_limit`
- `--split_test_size`
- `--split_random_state`

---

### 2. Single-Lead Mode

```bash
leadopt train \
  --preset medchem_quality_tier4 \
  --smiles "c1ccc(cc1)O" \
  --out_dir runs/ppo_single \
  --total_updates 50
```

---

### Resume Training

```bash
leadopt train \
  --preset medchem_quality_tier4 \
  --dataset dataset.csv \
  --smiles_col smiles \
  --out_dir runs/ppo_run \
  --resume runs/ppo_run/checkpoint_update_00090.pt
```

If `--resume` is omitted, training automatically resumes from:

```
out_dir/model_last.pt
```

if present.

---

## `leadopt generate` (Legacy PPO Workflow)

Generates molecules from a trained PPO checkpoint.

```bash
leadopt generate \
  --run_dir runs/ppo_run \
  --checkpoint model_best \
  --smiles "CCO" \
  --policy sample \
  --episodes 200 \
  --top_k 50
```

### Notes

- Output defaults to:

```
run_dir/generated_topk_<policy>.csv
```

- This subcommand intentionally mirrors the historical repository workflow.
- For reproducibility and environment consistency, prefer using explicit `--preset` when supported.

---

## `leadopt beam`

Deterministic beam search generation using a YAML preset (operators + scorer).

Beam-specific knobs are configured in the preset YAML under a `beam:` section:

```yaml
beam:
  complexity_weight: 0.05
  dock_drop_tolerance: null
  hard_constraint_filter: true
```

### Example

```bash
leadopt beam \
  --preset np_fragment_discovery \
  --smiles "CC(=O)NCc1ccccc1" \
  --beam_width 20 \
  --max_steps 8 \
  --per_state_action_limit 256 \
  --top_n 50 \
  --out_csv runs/beam.csv
```

### Outputs

- CSV file specified by `--out_csv`
- JSONL initialization pool file:
  - Same name as `--out_csv`
  - Suffix changed to `.pool.jsonl`
  - Used for downstream workflows

---

## Deprecated Legacy Flags (Hidden)

Historically, beam parameters were provided entirely via CLI flags.

These flags are still accepted for backward compatibility but are considered deprecated.  
Beam configuration should now reside in the preset YAML under `beam:`.

---

## Reproducibility Checklist (Recommended)

Each training run directory should contain:

- `preset_used.yaml`
- `run_manifest.json`
- `run_config.json`
- Checkpoints:
  - `model_best.pt`
  - `model_last.pt`
  - `checkpoint_update_*.pt`
- `vocab.json`
- `operators_sig.txt`

These artifacts ensure that experiments are reproducible and traceable.
