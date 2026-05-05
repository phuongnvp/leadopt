"""API types (stable, versioned dataclasses).

Phase 2.2:
- Field sets are now aligned with the public API plan (v1).
- Schema versioning is explicit via RunMetadata.api_schema_version.
- All objects provide conservative JSON-serialization helpers.

Design constraints:
- Avoid importing heavy deps (torch/rdkit) at module import time.
- Backwards compatible within the API (we only add fields; never remove in v1).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Literal, Optional

API_SCHEMA_VERSION: Literal["v1"] = "v1"


def _default_version() -> str:
    """Return leadopt+API schema version string, computed lazily."""
    try:
        import leadopt  # local import to avoid any import-time surprises

        pkg_version = getattr(leadopt, "__version__", "0.0.0+unknown")
    except Exception:
        pkg_version = "0.0.0+unknown"
    return f"{pkg_version}|api:{API_SCHEMA_VERSION}"


def _json_sanitize(obj: Any) -> Any:
    """Convert obj into JSON-serializable form.

    Policy:
    - dataclasses -> dict (recursively sanitized)
    - dict/list/tuple -> recursively sanitized
    - primitives pass through
    - everything else -> str(obj) as a conservative fallback

    This keeps academic reproducibility: results can always be logged,
    even when metadata contains non-JSON-native values.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj):
        return _json_sanitize(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return str(obj)


@dataclass(frozen=True, slots=True)
class _Serializable:
    """Mixin for stable return objects."""

    def to_dict(self) -> Dict[str, Any]:
        return _json_sanitize(asdict(self))

    def to_json(self, *, indent: int = 2, sort_keys: bool = True) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys)


# -----------------------------
# Core record and trace objects
# -----------------------------


@dataclass(frozen=True, slots=True)
class MoleculeRecord(_Serializable):
    """A molecule and its associated scores/metadata.

    smiles: canonical SMILES (canonicalization performed by engines / CLI)
    objective: scalar objective used for ranking/optimization
    components: optional named components (e.g., docking, qsar, penalties)
    metadata: free-form JSON-serializable metadata (step, provenance, etc.)
    """

    smiles: str
    objective: float
    components: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActionStep(_Serializable):
    t: int
    operator: str
    template: Optional[str]
    action_index: Optional[int]
    intermediate_smiles: str


@dataclass(frozen=True, slots=True)
class ActionTrace(_Serializable):
    steps: List[ActionStep]
    terminated: bool
    length: int


@dataclass(frozen=True, slots=True)
class RunMetadata(_Serializable):
    seed: int
    device: Optional[str] = None
    preset_name: Optional[str] = None
    preset_path: Optional[str] = None
    run_dir: Optional[str] = None

    # Versioning (stable contract)
    api_schema_version: Literal["v1"] = API_SCHEMA_VERSION
    version: str = field(default_factory=_default_version)

    # Free-form
    extra: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Result objects
# -----------------------------


@dataclass(frozen=True, slots=True)
class RunResult(_Serializable):
    lead: MoleculeRecord
    final: MoleculeRecord
    trace: ActionTrace
    metadata: RunMetadata


@dataclass(frozen=True, slots=True)
class BeamResult(_Serializable):
    lead: MoleculeRecord
    candidates: List[MoleculeRecord]
    metadata: RunMetadata
    artifacts: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenerateResult(_Serializable):
    lead: MoleculeRecord
    unique_count: int
    candidates: List[MoleculeRecord]
    metadata: RunMetadata
    artifacts: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainResult(_Serializable):
    run_dir: str
    best_checkpoint: Optional[str]
    last_checkpoint: Optional[str]
    checkpoints: List[str]
    train_summary: Dict[str, float]
    metadata: RunMetadata
    artifacts: Dict[str, str] = field(default_factory=dict)


__all__ = [
    "API_SCHEMA_VERSION",
    "MoleculeRecord",
    "ActionStep",
    "ActionTrace",
    "RunMetadata",
    "RunResult",
    "BeamResult",
    "GenerateResult",
    "TrainResult",
]
