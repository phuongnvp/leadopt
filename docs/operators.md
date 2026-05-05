# Operators in leadopt

This document defines the **frozen operator contract** for academic reproducibility.

Operators define the action space of the molecular environment. Each operator:

- Enumerates feasible `ActionInstance`s for a given molecule and constraint context
- Applies a selected `ActionInstance` to produce a new molecule
- Reports which parts of the **original** molecule were touched (for constraint locking, masking, and analysis)

> Operators are configured via YAML presets.  
> See `docs/presets.md` for configuration details.

---

# Core Types

## `ActionOperator`

An operator must implement:

- `enumerate_actions(mol, ctx) -> Sequence[ActionInstance]`
- `apply(mol, action) -> AppliedAction`
- `touched(mol, action) -> (touched_atoms, touched_bonds)`  
  (in **original index space**)

Operators may optionally implement:

- `is_feasible(mol, action, ctx) -> bool`

However, correctness must **not** rely solely on `is_feasible()`.  
Invalid operations must still be guarded in `apply()`.

---

## `ActionInstance`

An `ActionInstance` represents a single edit candidate.

Fields:

- `operator: str`  
  Operator identity
- `site: Tuple[int, ...]`  
  Operator-defined indices (atoms and/or bonds)
- `template: Optional[str]`  
  Optional label (e.g., swap id)
- `payload: Dict[str, Any]`  
  Operator-specific parameters (should be JSON-friendly)

### Important Notes

- `site` is **not globally standardized**.
- Some operators use atom indices.
- Some use bond indices.
- Some use mixed tuples.
- Not all indices in `site` are necessarily “touched”.

Do not assume `site` equals the touched set.

---

## `AppliedAction`

The result of applying an action:

- `mol: Chem.Mol`  
  A sanitized, valid molecule
- `action: ActionInstance`  
  The applied action
- `touched_atoms: Set[int]`
- `touched_bonds: Set[int]`

---

# Frozen Contract (Reproducibility Guarantees)

These invariants must not change without a versioned contract update.

---

## 1. Deterministic Enumeration

`enumerate_actions()` must be deterministic for a given `(mol, ctx)`:

- No hidden randomness
- Stable ordering
- No dependence on Python hash seed or object ids

`ActionInstance.stable_sort_key()` is the canonical ordering key and **must** be used (directly or indirectly).

---

## 2. Stable Operator Signatures (Resume Safety)

Training and CLI workflows compute an operator signature from `repr(op)`.

Therefore:

- `repr(op)` must be stable across processes for the same configuration
- It must not include memory addresses or nondeterministic fields

Operators inherit a stable `__repr__` from `ActionOperator`, based on a JSON-stable configuration view.

Changing `repr()` changes the operator signature and may invalidate training resumes.

---

## 3. Payload Discipline

`ActionInstance.payload` must be:

- Deterministic
- JSON-serializable (strongly preferred)

Avoid:

- Unordered sets
- RDKit objects
- Memory addresses
- Process-specific identifiers

If non-JSON objects are unavoidable:

- Their `repr()` must be stable
- They must not encode runtime-specific state

---

## 4. `apply()` Invariants

`apply(mol, action)` must:

- **Not mutate** the input `mol`
- Either:
  - Return an `AppliedAction` with a valid, sanitized molecule
  - Raise `ActionError` for invalid or infeasible operations

Silent failure or partial mutation is not allowed.

---

## 5. `touched()` Invariants (Frozen Index Space)

`touched(mol, action)` and `AppliedAction.touched_*` must be expressed in the **original molecule index space**.

This means:

- Atom indices refer to the pre-action molecule
- Bond indices refer to the pre-action molecule
- Newly created atoms or bonds must **not** appear in the touched sets

Touched sets are used for:

- Conservative constraint locking
- Action masking
- Reproducible logging and analysis

Violating this invariant breaks reproducibility guarantees.

---

## 6. Consistency Between `touched()` and `apply()`

`apply()` must return touched sets consistent with `touched()`:

```
AppliedAction.touched_atoms == operator.touched(mol, action)[0]
AppliedAction.touched_bonds == operator.touched(mol, action)[1]
```

Mismatch is a contract violation.

---

# Operator-Set Presets

leadopt ships several action-space presets for different editing regimes:

- `leadopt/presets/scaffold_hop.yaml`
- `leadopt/presets/linker.yaml`
- `leadopt/presets/decomplexify.yaml`

Each preset selects a constrained set of operators suited to a specific medicinal chemistry task.

These presets define:

- Which operators are active
- Operator configuration parameters
- Constraint and scorer combinations

---

# SMIRKS Libraries (Tier 3.2)

`SmirksLibraryOperator` provides named, versioned, auditable functional-group transformations via a curated one-reactant SMIRKS library.

Default library:

```
leadopt/data/smirks/medchem_smirks_v1.yaml
```

Each action payload includes:

- `library_version`
- `transform_name`
- `transform_index`
- `smirks` string

### Determinism Guarantees

- Library transforms are enumerated in file order
- Substructure matches are sorted deterministically
- Action payloads are stable and reproducible

---

# Design Philosophy

The operator system is built for:

- Deterministic action-space construction
- Resume-safe training
- Auditable edit traces
- Reproducible scientific experiments

Any modification to operator behavior must preserve:

- Determinism
- Stable `repr()`
- Stable touched index semantics
- JSON-stable payload structure

Changes that affect these properties require explicit versioning and documentation.
