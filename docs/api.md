# leadopt Python API (v1)

This document describes the stable, importable Python API that mirrors the CLI workflows:

- `leadopt train` → `leadopt.api.train(...)`
- `leadopt generate` → `leadopt.api.generate(...)`
- `leadopt beam` → `leadopt.api.beam(...)`
- `leadopt run` → `leadopt.api.run(...)`

The API is designed for:

- notebooks and research pipelines,
- reproducible runs (optional artifact writing),
- stable return objects (versioned dataclasses),
- minimal divergence from CLI semantics (engines are shared between CLI and API).

---

## Installation

Install from source (editable mode):

```bash
pip install -e .
```

Optional dependencies:

- **RDKit** — required for chemistry operations.
- **torch** — required for PPO training and policy-based generation.

---

## API Overview

```python
from leadopt.api import run, beam, train, generate
```

Each function corresponds directly to a CLI workflow and shares the same underlying engine implementation.

---

## Shared I/O Behavior

All API entrypoints accept the following common arguments:

- `run_dir: str | Path | None`
- `write_artifacts: bool`
- `verbose: bool` (reserved; engines are quiet by default)

### Modes

**Pure in-memory mode**

- `run_dir=None` or `write_artifacts=False`
- Returns dataclass result objects
- Does not write auxiliary files (except essential training outputs such as checkpoints and vocabularies)

**Artifact-writing mode**

- `run_dir="..."` and `write_artifacts=True`
- Writes the same core files as the corresponding CLI workflow

---

## Stable Return Objects

All results are dataclasses and JSON-serializable via:

```python
from dataclasses import asdict
import json

json.dumps(asdict(result), default=str)
```

### Common Data Structures

**MoleculeRecord**

- `smiles`
- `objective`
- optional `components`
- `metadata`

**RunMetadata**

- `seed`
- `device`
- `preset_name`
- `preset_path`
- `run_dir`
- `version`
- `api_schema_version`
- `extra`

### Result Types

**RunResult**

- `lead`
- `final`
- `trace`
- `metadata`

**BeamResult**

- `lead`
- `candidates`
- `metadata`
- `artifacts`

**GenerateResult**

- `lead`
- `unique_count`
- `candidates`
- `metadata`
- `artifacts`

**TrainResult**

- `run_dir`
- `checkpoints`
- `best_checkpoint`
- `last_checkpoint`
- `train_summary`
- `metadata`
- `artifacts`

---

## Presets

All workflows are configured using YAML presets.

The `preset=` argument accepts either:

- A shipped preset name (e.g., `medchem_quality_tier4`, `np_fragment_discovery`), or
- A path to a preset YAML file.

Presets define:

- operators,
- scorer,
- constraints,
- and (for beam search) a `beam:` configuration section.

---

# Examples

---

## 1. Run a Single Rollout (`leadopt run`)

```python
from leadopt.api import run

res = run(
    preset="medchem_quality_tier4",
    smiles="CCO",
    steps=3,
    seed=0,
    run_dir=None,            # pure in-memory
    write_artifacts=False,
)

print(res.final.smiles, res.final.objective)
```

To write CLI-compatible artifacts (e.g., `result.json`):

```python
res = run(
    preset="medchem_quality_tier4",
    smiles="CCO",
    steps=1,
    seed=0,
    run_dir="runs/api_run_demo",
    write_artifacts=True,
)

print(res.metadata.run_dir)
```

---

## 2. Beam Search (`leadopt beam`)

Preset-driven defaults:

If `beam_width`, `max_steps`, `per_state_action_limit`, or `top_n` are `None`, the engine resolves them from the preset’s `beam:` configuration (with CLI fallbacks).

```python
from leadopt.api import beam

br = beam(
    preset="np_fragment_discovery",
    smiles="CC(=O)NCc1ccccc1",
    seed=0,
    beam_width=2,
    max_steps=1,
    top_n=5,
    run_dir="runs/api_beam_demo",
    write_artifacts=True,
)

print("Top:", br.candidates[0].smiles, br.candidates[0].objective)
print("Artifacts:", br.artifacts)
```

---

## 3. Train PPO Then Generate (Pipeline)

```python
from pathlib import Path
from leadopt.api import train, generate

run_dir = Path("runs/api_train_demo")

tr = train(
    preset="medchem_quality_tier4",
    smiles="CCO",
    run_dir=run_dir,
    seed=0,
    total_updates=1,     # small smoke run
    eval_every=1,
    save_every=1,
    write_artifacts=False,  # keeps checkpoints/vocab but prunes optional logs
)

ckpt = tr.last_checkpoint or tr.best_checkpoint
ckpt_token = Path(ckpt).name if ckpt else "model_last.pt"

gr = generate(
    preset="medchem_quality_tier4",
    run_dir=run_dir,
    checkpoint=ckpt_token,
    smiles="CCO",
    seed=0,
    episodes=8,
    top_k=10,
    policy="sample",
    write_artifacts=False,
)

print("Unique:", gr.unique_count)
print("Top:", gr.candidates[0].smiles, gr.candidates[0].objective)
```

---

## 4. Determinism

For small runs, determinism can be checked by reusing the same seed.

Beam search should be deterministic for a given:

- seed,
- preset,
- input SMILES,
- and beam configuration.

```python
from leadopt.api import beam

b1 = beam(
    preset="np_fragment_discovery",
    smiles="CCO",
    seed=123,
    beam_width=2,
    max_steps=1,
    top_n=2,
)

b2 = beam(
    preset="np_fragment_discovery",
    smiles="CCO",
    seed=123,
    beam_width=2,
    max_steps=1,
    top_n=2,
)

assert b1.candidates[0].smiles == b2.candidates[0].smiles
```

---

## JSON Export

```python
import json
from dataclasses import asdict
from leadopt.api import run

r = run(
    preset="medchem_quality_tier4",
    smiles="CCO",
    steps=1,
    seed=0,
    write_artifacts=False,
)

json_payload = json.dumps(asdict(r), default=str, indent=2)
print(json_payload[:200])
```

---

## Troubleshooting

- **Preset lookup fails**  
  Pass an explicit YAML path:

  ```python
  preset="leadopt/presets/medchem_quality_tier4.yaml"
  ```

- **Generation fails in legacy mode**  
  Prefer passing `preset=...` explicitly to ensure environment/operator compatibility.

- **CI / smoke runs**  
  Keep:
  - `total_updates=1` for training
  - `episodes <= 8` for generation
  - small beam widths and step counts

This keeps tests fast and deterministic.
