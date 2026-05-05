# Constraints

Constraints in **leadopt** return a _margin_ value:

- `margin > 0` → constraint satisfied
- `margin = 0` → boundary condition
- `margin < 0` → violated (magnitude reflects severity)

Constraints are evaluated through a `ConstraintSuite` and consumed by the `RewardComposer` as penalties (if configured in the preset).

---

## Constraint Architecture

- Individual constraint classes implement margin-based evaluation.
- `ConstraintSuite` aggregates multiple constraints.
- The `RewardComposer` can:
  - Penalize violations softly (via weighted margin),
  - Or enforce hard filtering (depending on preset configuration).

This design enables:

- Continuous penalty shaping,
- Explicit hard gating when required,
- Reproducible constraint behavior via preset definitions.

---

# Medchem Filter Constraints (Tier 1)

These are **opt-in constraints** intended to keep generated molecules chemically reasonable for standard medicinal chemistry workflows.

They are lightweight, interpretable, and suitable for general-purpose lead optimization tasks.

---

## `ChargeConstraint`

Bounds the total formal charge of a molecule.

Typical configuration:

- `min_charge = -1`
- `max_charge = +1`

Use case:

- Prevents extreme charge states,
- Maintains compatibility with drug-like chemical space.

---

## `ElementConstraint`

Ensures that only allowed elements appear in generated molecules.

Default allowed elements:

```
H, C, N, O, S, F, Cl, Br, I, P
```

Use case:

- Restricts chemical space to drug-like organic chemistry,
- Avoids metals or exotic atoms unless explicitly enabled.

---

## `RingCountConstraint`

Caps (and optionally lower-bounds) the total ring count.

Use case:

- Prevents overly complex polycyclic structures,
- Controls synthetic accessibility heuristically,
- Avoids unrealistic topologies.

---

## `HBDHBAConstraint`

Caps (and optionally lower-bounds) hydrogen bond donors (HBD) and acceptors (HBA).

Use case:

- Maintains Lipinski-style drug-likeness constraints,
- Controls polarity and permeability indirectly.

---

## `ReactiveGroupConstraint`

Bans a curated SMARTS blacklist of reactive or unstable functional groups (Tier 1 minimal list).

Examples include:

- Highly reactive electrophiles,
- Labile or unstable moieties unsuitable for medicinal chemistry workflows.

### Reproducibility

This constraint is versioned via:

```
library_version
```

in its metadata to ensure:

- Explicit tracking of blacklist revisions,
- Reproducible behavior across releases.

---

# Design Notes

- All constraints operate on parsed molecular structures.
- Margins allow flexible integration into reward shaping.
- Hard filtering (when enabled) removes violating candidates entirely.
- Constraint configuration is defined in YAML presets to ensure experiment traceability.

For strict reproducibility, always archive:

- `preset_used.yaml`
- `run_manifest.json`
- Any constraint metadata fields (e.g., `library_version`)
