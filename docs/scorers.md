# Scorers in leadopt

This document defines the **frozen scorer contract** for academic reproducibility.

A scorer maps an RDKit molecule to a `ScoringResult` and must be:

- Deterministic whenever possible (or seed-controlled)
- Failure-safe (must never raise; returns `valid = False`)
- Direction-standardized (**higher is better**)

---

# ScoringResult Contract

All scorers return a `ScoringResult` with the following fields:

### `objective: float`

- Standardized direction: **higher is better**
- If a raw score is naturally “lower is better” (e.g., docking energy), the scorer must convert it.

---

### `components: Dict[str, float]`

- Per-component breakdown for interpretability.
- Recommended to include:
  - `objective` (duplicate for clarity)
  - `compute_cost_units` (if applicable)
  - `compute_cost_s` (if measured)

---

### `metadata: Dict[str, Any]`

- Must be JSON-serializable.
- Must include the scorer identity from `scorer_metadata()`.
- May include additional fields (e.g., cache status, hashes, protocol details).

---

### `valid: bool`

- `True` if the score is meaningful.
- `False` if scoring failed.

---

### `fail_reason: Optional[str]`

- Required when `valid = False`.
- Must follow the failure taxonomy defined below.

---

# Objective Direction Convention

All scorers must follow:

> **Higher objective value = better molecule**

Example (docking):

- Raw docking energy is typically negative and “lower is better”.
- Store raw value in:
  ```
  components["docking_energy"]
  ```
- Convert:
  ```
  objective = -docking_energy
  ```

This guarantees consistent optimization behavior across all workflows.

---

# Failure Taxonomy (Phase 5.4)

Scorers must never raise exceptions during scoring.  
Failures are encoded using:

- `valid = False`
- `fail_reason` with one of the following prefixes:

### `input:<reason>`

Examples:

- invalid molecule
- missing receptor file
- malformed configuration

### `budget:<reason>`

Examples:

- docking budget exceeded

### `engine:<reason>`

Examples:

- backend unavailable
- external tool failure
- engine not implemented

### `exception:<ExceptionType>`

Unexpected exception caught internally.

This uniform taxonomy enables consistent experiment-wide failure analysis.

---

# Compute-Cost Semantics (Phase 5.1)

Some scorers (e.g., docking) are computationally expensive.

The compute-cost contract is:

### Preferred field:

```
metadata["compute_cost_units"]
```

Unitless cost (e.g., number of docking calls) used for reward penalties.

### Legacy alias:

```
metadata["compute_cost"]
```

Must equal `compute_cost_units` when present.

### Optional:

```
metadata["compute_cost_s"]
```

Wall time in seconds (informational; last-resort fallback).

Reward shaping reads cost in this order:

1. `compute_cost_units`
2. `compute_cost`
3. `compute_cost_s`

---

# Caching Policy (DockingScorer) — Phase 5.2

Docking cache entries are keyed by a deterministic payload including:

- `cache_schema_version`
- `scorer_version`
- canonical SMILES
- receptor hash
- protocol
- box
- params
- engine name + engine_version
- seed

Cache hits are accepted only if:

- Stored `key_payload` exactly matches the current payload.

This protects against:

- Cache corruption
- Schema drift
- Engine version changes
- Protocol changes

---

# Scorer Metadata Requirements

Every scorer must implement:

```
scorer_metadata() -> Dict[str, Any]
```

Must include:

- `name`
- `type` (e.g., `qsar`, `mpo`, `docking`)
- `version`
- Scorer-specific reproducibility fields

All scorer metadata is written to `run_manifest.json`.

---

# Included Scorers

---

## QSARScorer

- Deterministic RDKit-based properties and/or user-supplied models.
- Logs component breakdown.
- Failure-safe and taxonomy-compliant.

---

## RealQSARScorer (`type: qsar_real`)

User-supplied pickled model (`model.pkl`) used as a real QSAR backend.  
No bundled models are provided.

### Supported Input Modes

Configured via:

```
scoring.params.model.input_mode
```

### 1) `fingerprint` (default)

- leadopt computes deterministic Morgan fingerprints.
- Calls:
  ```
  model.predict(X)
  ```
  where `X` is a NumPy array.
- Reproducibility:
  - `model_sha256` recorded in metadata and cache keys
  - `feature_sha256` derived from fingerprint config

### 2) `smiles`

- leadopt canonicalizes SMILES.
- Calls:
  ```
  model.predict(smiles_list)
  ```
- Descriptor determinism is the user’s responsibility.

Example preset:

```
leadopt/presets/qsar_real.yaml
```

---

## MPOScorer

- Multi-property objective via aggregation (e.g., weighted sum).
- Logs full per-property components.
- Failure-safe and taxonomy-compliant.

---

## DockingScorer

Supported protocols:

- `standard`
- `aligned_local` (Phase 6.1)

Supported engines:

- `mock` (CI/development)
- `vina_cli` (real backend via subprocess)

Features:

- Disk cache
- Budget control
- Failure-safe scoring
- Standardized objective:
  ```
  objective = -docking_energy
  ```

---

### Vina CLI Backend (`engine: vina_cli`) Reproducibility Checklist

- Provide receptor in **PDBQT** format.
- Prefer deterministic settings:
  - `params.cpu: 1`
  - `OMP_NUM_THREADS=1`
- Seed is logged and used when supported.
- Metadata should include:
  - Engine path and version
  - Optional `engine_sha256`
  - Full CLI arguments
  - Bounded stdout/stderr capture

Optional integration test:

- Enable with:
  ```
  LEADOPT_RUN_VINA_TESTS=1
  ```
- Provide:
  ```
  LEADOPT_VINA_RECEPTOR=/abs/path/to/receptor.pdbqt
  ```
- Optional:
  ```
  LEADOPT_VINA_BINARY=/abs/path/to/vina
  ```

---

## `aligned_local` Protocol Details

Designed for pose-anchored local docking:

1. Align generated ligand to reference co-ligand (SDF).
2. Perform local refinement in a tight box.

Phase 6.1 guarantees:

- Deterministic conformer generation
- MCS-based alignment
- Optional derived local box (`anchor.derive_box: true`)
- Engine receives:
  - `local_only = true`
  - aligned initial pose
- Cache key includes `_initial_pose_hash`

If `derive_box = true`, the user-provided `box` field is overridden.

---

# CompositeScorer (Multi-Objective) — `type: composite`

A wrapper that evaluates multiple sub-scorers and aggregates them into a single scalar objective.

Goals:

- Preserve single-scalar environment contract
- Preserve full per-objective metadata
- Remain failure-safe

### YAML Shape (Phase 9 MVP)

```yaml
scoring:
  type: composite
  params:
    aggregation:
      mode: weighted_sum
      weights:
        qsar: 1.0
        docking: 0.2
    scorers:
      - name: qsar
        type: qsar_real
        params: { ... }
      - name: docking
        type: DockingScorer
        params: { ... }
```

### Aggregation (MVP)

```
objective = sum_i weights[name_i] * sub_objective_i
```

### Logging Expectations

- `components["obj:<name>"] = sub_objective`
- `metadata["sub_results"]` includes:
  - objective
  - valid
  - fail_reason
  - selected metadata
- `metadata["aggregation"]` includes:
  - mode
  - weights

> Note: This section documents the intended contract; implementation proceeds in Phase 9.A1+.

---

# Design Principles

All scorers must preserve:

- Determinism (or seed control)
- Failure safety
- JSON-serializable metadata
- Stable objective direction
- Explicit versioning

Any change affecting these guarantees requires versioning and documentation.
