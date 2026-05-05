# leadopt YAML Presets

This document defines the YAML experiment configuration format used in **leadopt** for reproducible molecular RL workflows.

Presets are the **single source of truth** for experiment configuration.  
Every CLI and API workflow loads a preset and constructs:

- Operators (action space)
- Constraints
- Scorer
- Reward composer
- Training configuration
- Optional beam-search configuration

> **Scorer contract:** see `docs/scorers.md`  
> **Operator contract:** see `docs/operators.md`

---

# Reproducibility Checklist

Every run directory should contain:

- `preset_used.yaml` (exact copy of the loaded preset)
- `run_manifest.json`, including:
  - unified seed
  - scorer metadata
  - constraint metadata
  - reward settings
  - environment/training configuration
  - git commit hash (if enabled)
  - docking engine/protocol/cache fields (if applicable)

Presets are archived verbatim to ensure experiments can be reconstructed exactly.

---

# Design Goals

- Single-file experiment configuration (YAML)
- Exact preset snapshot stored per run
- Manifest logging for reproducibility
- Stable operator-set configuration (resume safety)
- Deterministic scorer and constraint behavior

---

# Preset Structure

A preset YAML must contain the following top-level sections:

```yaml
preset_version: 1
name: docking

actions:
  operators:
    - type: PruneTerminal
      params: {}
  max_steps: 10
  rules:
    ban_motifs: true
    max_mw: 650.0

constraints:
  - type: SimilarityConstraint
    params:
      lead_smiles: "CCO"
      min_sim: 0.25

scoring:
  type: LegacyFunctionScorer
  params:
    function: "leadopt.score.docking:example_docking_score"
    metadata:
      receptor: "path/to/receptor.pdbqt"
      grid_box: [0, 0, 0, 20, 20, 20]
      reference_ligand: "path/to/ref.sdf"

reward:
  mode: terminal
  gamma: 0.99
  step_penalty: 0.01
  constraint_penalty_weight: 1.0
  compute_cost_weight: 0.0
  bonus: 0.0

training:
  algorithm: ppo
  total_timesteps: 200000
  gamma: 0.99
  seed: 0

beam:
  complexity_weight: 0.05
  dock_drop_tolerance: null
  hard_constraint_filter: true

logging:
  save_preset_yaml: true
  log_git_commit: true
  log_scorer_metadata: true
  log_constraint_metadata: true
```

---

# Section Breakdown

## `preset_version`

Version tag for future schema evolution.  
Used to prevent silent breaking changes.

---

## `actions`

Defines the action space and environment editing behavior.

### `operators`

List of operators with configuration parameters.

Operator sets must remain stable within a run (resume safety).

See `docs/operators.md` for the frozen operator contract.

### `max_steps`

Maximum environment steps per episode.

### `rules`

Optional environment-level gating rules (e.g., motif bans, MW caps).

---

## `constraints`

List of constraint definitions.

Each entry specifies:

- `type`
- `params`

Constraints are evaluated via `ConstraintSuite` and may be used for:

- soft penalties
- hard filtering (depending on configuration)

---

## `scoring`

Defines the scoring function and metadata.

The scorer must conform to the frozen `ScoringResult` contract.

Metadata fields are stored in the run manifest for reproducibility.

---

## `reward`

Defines reward composition behavior.

Key fields:

- `mode` — e.g., `terminal`
- `gamma`
- `step_penalty`
- `constraint_penalty_weight`
- `compute_cost_weight`
- `bonus`

These parameters directly affect learning dynamics and must be logged.

---

## `training`

Defines PPO-specific parameters.

Common fields:

- `algorithm`
- `total_timesteps`
- `gamma`
- `seed`

Training configuration is stored in `run_manifest.json`.

---

## `beam` (Optional)

Used by `leadopt beam`.

Fields:

- `complexity_weight`  
  Augmented objective:  
  `augmented = objective - complexity_weight * complexity`

- `dock_drop_tolerance`  
  Optional constraint: require  
  `objective >= start_objective - tolerance`

- `hard_constraint_filter`  
  If true, reject candidates with any negative constraint margin.

If CLI flags are omitted, values are resolved from this section.

---

## `logging`

Controls artifact logging:

- `save_preset_yaml`
- `log_git_commit`
- `log_scorer_metadata`
- `log_constraint_metadata`

---

# Operator-Set Presets

leadopt ships predefined operator-set presets.

These define the action space and editing regime.

---

## `scaffold_hop`

Preset file:

```
leadopt/presets/scaffold_hop.yaml
```

Operators:

- `PruneTerminal`
- `AzaScanAromatic`
- `BioisostereSwap`

Intended use:

Scaffold-level edits while maintaining similarity to a reference.

---

## `linker`

Preset file:

```
leadopt/presets/linker.yaml
```

Operators:

- `LinkerInsertCH2`
- `LinkerDeleteCH2`

Intended use:

Linker length tuning via CH₂ insertion/deletion.

---

## `decomplexify`

Preset file:

```
leadopt/presets/decomplexify.yaml
```

Operators:

- `DeleteSubtree`
- `PruneTerminal`

Intended use:

Simplify structures by pruning sidechains under size constraints.

---

# DockingScorer (Modern)

`DockingScorer` standardizes docking to the scoring contract.

Properties:

- Component: `docking_energy` (raw, typically negative)
- Objective convention:  
  `objective = -docking_energy`
- Failure-safe behavior: missing receptor or engine returns `valid=false`
- Supports deterministic `engine: mock` for CI and development

Example preset:

```
leadopt/presets/docking_modern.yaml
```

Key fields:

- `receptor_path`
- `box.center`
- `box.size`
- `cache_dir`
- `budget`

---

# Docking Protocol: `aligned_local` (Phase 6.1)

`aligned_local` is intended for pose-anchored local docking:

1. Align generated ligand to a reference co-ligand (SDF)
2. Perform local-only refinement in a tight docking box

Phase 6.1 implements:

- Deterministic conformer generation
- MCS-based alignment
- Optional derived local box
- Engine request fields:
  - `local_only`
  - `initial_pose_molblock`
- Cache-key hashing including protocol payload

Example preset:

```
leadopt/presets/docking_aligned_local.yaml
```

Additional fields:

- `reference_ligand_path`
- `reference_conformer`
- `alignment`
- `anchor`
  - `derive_box`
  - `box_padding_A`
  - `box_min_A`
- `local_opt`

Note:

A `box` field is still required for validation.  
If `anchor.derive_box: true`, the scorer overrides it with a derived local box.

---

# Phase 7.6 Notes (Complexity Shaping + Environment Correctness)

This section records behavior required for reproducible RL runs.

---

## Complexity Shaping

Optional shaping term:

- Environment computes:
  - `complexity_prev`
  - `complexity_curr`
  - `complexity_delta` (positive when complexity decreases)

If `complexity_weight != 0`:

```
complexity_term = complexity_weight * complexity_delta
```

This term appears in:

```
info["reward_breakdown"]
```

for auditability.

---

## GraphEnvironment Action Masking + Caching

`available_actions()` returns:

```
(actions, mask)
```

Guarantees:

- `mask[i] == True` iff action passes gating
- Gating uses `filter_allowed_with_applied`
- Cached per:
  - `(state.step, canonical_smiles(state.mol))`
- Cache refreshes automatically after edits

This prevents stale actions referencing invalid atom indices.

---

## Result Storage Contract

Scoring results are stored in:

```
info["_result"]
```

(and mirrored into `state.info["_result"]`)

This avoids requiring an `env._result` attribute and improves resumability.

---

## Regression Tests (Phase 7.6)

Behavior is locked by:

- `tests/test_env_action_mask_cache_phase76.py`
- `tests/test_env_result_storage_and_complexity_term_phase76.py`

---

## Minimal CLI Smoke Run (Docking)

```bash
mkdir -p data/receptors

python - <<'PY'
from pathlib import Path
Path("data/receptors/example_receptor.pdbqt").write_text("RECEPTOR\n", encoding="utf-8")
print("wrote receptor stub")
PY

leadopt train \
  --preset leadopt/presets/docking_modern.yaml \
  --out_dir runs/_dock_smoke \
  --smiles CCO \
  --total_updates 1 \
  --eval_every 9999 \
  --save_every 9999
```

This validates preset loading, environment wiring, and docking scorer integration.
