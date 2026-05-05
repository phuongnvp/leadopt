from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .base import Scorer
from .types import ScoringResult

_QSAR_REAL_CACHE_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class _QSARRealCacheEntry:
    key_sha256: str
    key_payload: Dict[str, Any]
    objective: float


class _QSARRealCache:
    """Simple, deterministic disk cache: one JSON file per key hash."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)

    def _path_for_key(self, key_sha256: str) -> Path:
        return self.cache_dir / f"{key_sha256}.json"

    def get(self, key_sha256: str) -> Optional[_QSARRealCacheEntry]:
        p = self._path_for_key(key_sha256)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return _QSARRealCacheEntry(
                key_sha256=str(d["key_sha256"]),
                key_payload=dict(d["key_payload"]),
                objective=float(d["objective"]),
            )
        except Exception:
            return None

    def put(self, entry: _QSARRealCacheEntry) -> None:
        _ensure_dir(self.cache_dir)
        p = self._path_for_key(entry.key_sha256)
        d = {
            "key_sha256": entry.key_sha256,
            "key_payload": entry.key_payload,
            "objective": float(entry.objective),
        }
        p.write_text(json.dumps(d, sort_keys=True, indent=2), encoding="utf-8")


def _load_model(path: Path) -> Any:
    """Load a pickled model. Failure is handled by the caller."""
    # Prefer standard pickle. Optionally fall back to joblib if available.
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            pass
    try:
        import joblib  # type: ignore

        return joblib.load(str(path))
    except Exception as e:
        raise e


def _morgan_fingerprint_array(
    mol: Chem.Mol,
    *,
    radius: int,
    n_bits: int,
    use_chirality: bool,
    use_features: bool,
) -> np.ndarray:
    """Deterministic RDKit Morgan fingerprint to numpy array (shape=(n_bits,))."""
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol,
        radius=int(radius),
        nBits=int(n_bits),
        useChirality=bool(use_chirality),
        useFeatures=bool(use_features),
    )
    arr = np.zeros((int(n_bits),), dtype=np.int8)
    # RDKit fills numpy array in-place.
    from rdkit.DataStructs import (
        ConvertToNumpyArray,  # local import to keep module load light
    )

    ConvertToNumpyArray(fp, arr)
    # Many sklearn models expect float input.
    return arr.astype(np.float32, copy=False)


@dataclass
class RealQSARScorer(Scorer):
    """Real QSAR scorer using a user-supplied pickled model.

    Two supported model input modes:

    - fingerprint: leadopt computes a deterministic Morgan fingerprint and calls model.predict(X)
      where X is a numpy array of shape (n_samples, n_bits).
    - smiles: leadopt canonicalizes SMILES and calls model.predict(smiles_list) where smiles_list
      is a list[str]. Descriptor choice is entirely in the user's model pipeline.

    Notes
    -----
    * SECURITY: Pickle can execute code. Only load trusted model files.
    * Objective direction: higher predictions are treated as better.
    """

    # Model configuration
    model_path: str = ""
    input_mode: str = "fingerprint"  # fingerprint | smiles

    # Fingerprint configuration (used only when input_mode=fingerprint)
    features_kind: str = "morgan"
    features_radius: int = 2
    features_n_bits: int = 2048
    features_use_chirality: bool = True
    features_use_features: bool = False

    # Cache configuration
    cache_enabled: bool = True
    cache_dir: str = ".leadopt_cache/qsar_real"

    # Allow YAML override while keeping the Scorer base default.
    fail_objective: float = Scorer.fail_objective

    # Stable version string for reproducibility artifacts.
    version: str = "0"

    extra_metadata: Optional[Dict[str, Any]] = None

    # Runtime state (not part of dataclass init)
    _model: Any = None
    _model_sha256: Optional[str] = None
    _cache: Optional[_QSARRealCache] = None
    _calls_made: int = 0
    _cache_hits: int = 0

    def __post_init__(self) -> None:
        # Lazily load model on first score() to keep construction failure-safe.
        if self.cache_enabled:
            self._cache = _QSARRealCache(self.cache_dir)

    def _feature_config(self) -> Dict[str, Any]:
        return {
            "kind": str(self.features_kind),
            "radius": int(self.features_radius),
            "n_bits": int(self.features_n_bits),
            "use_chirality": bool(self.features_use_chirality),
            "use_features": bool(self.features_use_features),
        }

    def _feature_sha256(self) -> Optional[str]:
        if str(self.input_mode).strip().lower() != "fingerprint":
            return None
        d = self._feature_config()
        return _sha256_text(json.dumps(d, sort_keys=True))

    def scorer_metadata(self) -> Dict[str, Any]:
        rdkit_version = getattr(Chem.rdBase, "rdkitVersion", "unknown")
        md: Dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "type": "qsar_real",
            "input_mode": str(self.input_mode).strip().lower(),
            "model_path": str(self.model_path),
            "model_sha256": self._model_sha256,
            "rdkit_version": rdkit_version,
        }
        feat_hash = self._feature_sha256()
        if feat_hash is not None:
            md["feature_config"] = self._feature_config()
            md["feature_sha256"] = feat_hash
        if self.extra_metadata:
            md.update(dict(self.extra_metadata))
        return md

    def _ensure_model_loaded(self) -> Tuple[bool, Optional[str]]:
        """Ensure model is loaded; returns (ok, fail_reason)."""
        if self._model is not None:
            return True, None
        p = Path(str(self.model_path)).expanduser()
        if not p.exists():
            return False, "input:model_missing"
        try:
            self._model_sha256 = _sha256_file(p)
        except Exception:
            # Hashing failure should not prevent scoring.
            self._model_sha256 = None
        try:
            self._model = _load_model(p)
        except Exception:
            self._model = None
            return False, "input:model_load_failed"
        if not hasattr(self._model, "predict"):
            self._model = None
            return False, "input:model_load_failed"
        return True, None

    def _key_payload_and_hash(
        self, canonical_smiles: str
    ) -> Tuple[Dict[str, Any], str]:
        rdkit_version = getattr(Chem.rdBase, "rdkitVersion", "unknown")
        key_payload: Dict[str, Any] = {
            "cache_schema_version": int(_QSAR_REAL_CACHE_SCHEMA_VERSION),
            "scorer_version": str(self.version),
            "smiles": str(canonical_smiles),
            "input_mode": str(self.input_mode).strip().lower(),
            "model_sha256": self._model_sha256,
            "rdkit_version": rdkit_version,
        }
        feat_hash = self._feature_sha256()
        if feat_hash is not None:
            key_payload["feature_sha256"] = feat_hash
            key_payload["feature_config"] = self._feature_config()
        key_text = json.dumps(key_payload, sort_keys=True)
        return key_payload, _sha256_text(key_text)

    def score(
        self, mol: Any, context: Optional[Dict[str, Any]] = None
    ) -> ScoringResult:
        self._calls_made += 1
        t0 = time.time()
        compute_cost_units = 0.0
        compute_cost_s = 0.0

        try:
            if mol is None:
                return ScoringResult(
                    objective=float(self.fail_objective),
                    components={"objective": float(self.fail_objective)},
                    valid=False,
                    fail_reason="input:mol_is_none",
                    metadata={**self.scorer_metadata(), "exception_type": None},
                )

            ok, fail_reason = self._ensure_model_loaded()
            if not ok:
                md = {**self.scorer_metadata()}
                # Best-effort smiles
                try:
                    md["smiles"] = Chem.MolToSmiles(mol, isomericSmiles=True)
                except Exception:
                    md["smiles"] = None
                if context:
                    md["context"] = dict(context)
                md["cache_hit"] = False
                md["compute_cost_s"] = 0.0
                md["compute_cost_units"] = 0.0
                md["compute_cost"] = 0.0
                md["calls_made"] = int(self._calls_made)
                md["cache_hits"] = int(self._cache_hits)
                return ScoringResult(
                    objective=float(self.fail_objective),
                    components={"objective": float(self.fail_objective)},
                    valid=False,
                    fail_reason=str(fail_reason),
                    metadata=md,
                )

            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            if not canonical_smiles:
                return ScoringResult(
                    objective=float(self.fail_objective),
                    components={"objective": float(self.fail_objective)},
                    valid=False,
                    fail_reason="input:empty_smiles",
                    metadata={**self.scorer_metadata(), "smiles": canonical_smiles},
                )

            # Cache lookup
            key_payload, key_sha256 = self._key_payload_and_hash(canonical_smiles)
            cache_hit = False
            if self._cache is not None:
                entry = self._cache.get(key_sha256)
                if entry is not None and entry.key_payload == key_payload:
                    cache_hit = True
                    self._cache_hits += 1
                    objective = float(entry.objective)
                    md = {
                        **self.scorer_metadata(),
                        "smiles": canonical_smiles,
                        "cache_hit": True,
                        "compute_cost_s": 0.0,
                        "compute_cost_units": 0.0,
                        "compute_cost": 0.0,
                        "calls_made": int(self._calls_made),
                        "cache_hits": int(self._cache_hits),
                        "key_sha256": key_sha256,
                    }
                    if context:
                        md["context"] = dict(context)
                    return ScoringResult(
                        objective=float(objective),
                        components={"objective": float(objective)},
                        valid=True,
                        fail_reason=None,
                        metadata=md,
                    )

            # Cache miss: run model prediction.
            mode = str(self.input_mode).strip().lower()
            if mode == "fingerprint":
                if str(self.features_kind).strip().lower() != "morgan":
                    raise ValueError(
                        f"Unsupported features.kind '{self.features_kind}'"
                    )
                try:
                    x = _morgan_fingerprint_array(
                        mol,
                        radius=int(self.features_radius),
                        n_bits=int(self.features_n_bits),
                        use_chirality=bool(self.features_use_chirality),
                        use_features=bool(self.features_use_features),
                    )
                except Exception:
                    md = {**self.scorer_metadata(), "smiles": canonical_smiles}
                    md["cache_hit"] = False
                    md["compute_cost_s"] = 0.0
                    md["compute_cost_units"] = 0.0
                    md["compute_cost"] = 0.0
                    md["calls_made"] = int(self._calls_made)
                    md["cache_hits"] = int(self._cache_hits)
                    if context:
                        md["context"] = dict(context)
                    return ScoringResult(
                        objective=float(self.fail_objective),
                        components={"objective": float(self.fail_objective)},
                        valid=False,
                        fail_reason="input:featurization_failed",
                        metadata=md,
                    )

                X = x.reshape(1, -1)
                y = self._model.predict(X)  # type: ignore[attr-defined]
            elif mode == "smiles":
                y = self._model.predict([canonical_smiles])  # type: ignore[attr-defined]
            else:
                return ScoringResult(
                    objective=float(self.fail_objective),
                    components={"objective": float(self.fail_objective)},
                    valid=False,
                    fail_reason=f"input:unknown_input_mode:{self.input_mode}",
                    metadata={**self.scorer_metadata(), "smiles": canonical_smiles},
                )

            # Validate model output
            try:
                y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
            except Exception as e:
                raise TypeError("Model output is not array-like") from e
            if y_arr.size < 1:
                raise ValueError("Model output is empty")
            objective = float(y_arr[0])
            if not np.isfinite(objective):
                raise ValueError("Model output is not finite")

            compute_cost_s = float(time.time() - t0)
            compute_cost_units = 1.0

            # Store in cache (valid-only)
            if self._cache is not None:
                self._cache.put(
                    _QSARRealCacheEntry(
                        key_sha256=key_sha256,
                        key_payload=key_payload,
                        objective=float(objective),
                    )
                )

            md = {
                **self.scorer_metadata(),
                "smiles": canonical_smiles,
                "cache_hit": bool(cache_hit),
                "compute_cost_s": float(compute_cost_s),
                "compute_cost_units": float(compute_cost_units),
                "compute_cost": float(compute_cost_units),
                "calls_made": int(self._calls_made),
                "cache_hits": int(self._cache_hits),
                "key_sha256": key_sha256,
            }
            if context:
                md["context"] = dict(context)

            return ScoringResult(
                objective=float(objective),
                components={"objective": float(objective)},
                valid=True,
                fail_reason=None,
                metadata=md,
            )

        except Exception as e:
            compute_cost_s = float(time.time() - t0)
            # Only charge compute if we actually attempted work beyond trivial validation.
            if compute_cost_s > 1e-6:
                compute_cost_units = 1.0

            md = {**self.scorer_metadata()}
            md["exception_type"] = type(e).__name__
            md["exception_message"] = str(e)
            try:
                md["smiles"] = (
                    Chem.MolToSmiles(mol, isomericSmiles=True)
                    if mol is not None
                    else None
                )
            except Exception:
                md["smiles"] = None
            if context:
                md["context"] = dict(context)
            md["cache_hit"] = False
            md["compute_cost_s"] = float(compute_cost_s)
            md["compute_cost_units"] = float(compute_cost_units)
            md["compute_cost"] = float(compute_cost_units)
            md["calls_made"] = int(self._calls_made)
            md["cache_hits"] = int(self._cache_hits)

            return ScoringResult(
                objective=float(self.fail_objective),
                components={"objective": float(self.fail_objective)},
                valid=False,
                fail_reason=f"exception:{type(e).__name__}",
                metadata=md,
            )
