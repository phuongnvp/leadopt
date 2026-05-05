from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdMolAlign

from .base import Scorer
from .types import ScoringResult

_DOCKING_CACHE_SCHEMA_VERSION = 1


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
class _DockingCacheEntry:
    key_sha256: str
    key_payload: Dict[str, Any]
    docking_energy: float
    engine: str
    engine_version: str
    protocol: str
    receptor_hash: str
    box: Any
    params: Dict[str, Any]
    seed: Optional[int]


class _DockingCache:
    """Simple, deterministic disk cache: one JSON file per key hash."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)

    def _path_for_key(self, key_sha256: str) -> Path:
        return self.cache_dir / f"{key_sha256}.json"

    def get(self, key_sha256: str) -> Optional[_DockingCacheEntry]:
        p = self._path_for_key(key_sha256)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return _DockingCacheEntry(
                key_sha256=str(d["key_sha256"]),
                key_payload=dict(d["key_payload"]),
                docking_energy=float(d["docking_energy"]),
                engine=str(d["engine"]),
                engine_version=str(d["engine_version"]),
                protocol=str(d["protocol"]),
                receptor_hash=str(d["receptor_hash"]),
                box=d["box"],
                params=dict(d.get("params", {})),
                seed=(int(d["seed"]) if d.get("seed", None) is not None else None),
            )
        except Exception:
            # Failure-safe: treat corrupted entry as miss.
            return None

    def set(self, entry: _DockingCacheEntry) -> None:
        _ensure_dir(self.cache_dir)
        p = self._path_for_key(entry.key_sha256)
        payload = {
            "key_sha256": entry.key_sha256,
            "key_payload": entry.key_payload,
            "docking_energy": float(entry.docking_energy),
            "engine": entry.engine,
            "engine_version": entry.engine_version,
            "protocol": entry.protocol,
            "receptor_hash": entry.receptor_hash,
            "box": entry.box,
            "params": entry.params,
            "seed": entry.seed,
        }
        p.write_text(_stable_json(payload), encoding="utf-8")


def _stable_json(obj: Any) -> str:
    """Deterministic JSON representation for cache keys / hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _validate_box_dict(
    box: Dict[str, Any],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if not isinstance(box, dict):
        raise ValueError("box_not_dict")

    if "center" not in box or "size" not in box:
        raise ValueError("box_missing_center_or_size")

    center = box["center"]
    size = box["size"]

    if not isinstance(center, (list, tuple)) or len(center) != 3:
        raise ValueError("box_center_not_len3")
    if not isinstance(size, (list, tuple)) or len(size) != 3:
        raise ValueError("box_size_not_len3")

    c = (float(center[0]), float(center[1]), float(center[2]))
    s = (float(size[0]), float(size[1]), float(size[2]))
    return c, s


def _mol_has_3d(m: Chem.Mol) -> bool:
    return m.GetNumConformers() > 0 and m.GetConformer().Is3D()


def _ensure_3d_conformer(m: Chem.Mol, seed: int = 0) -> Chem.Mol:
    """Ensure mol has a 3D conformer deterministically.

    - If a conformer exists, return as-is.
    - Otherwise, embed + (light) optimize with deterministic seeding.

    This is used for aligned_local alignment prep. It is *not* intended as a full docking prep pipeline.
    """
    if _mol_has_3d(m):
        return m

    mm = Chem.AddHs(Chem.Mol(m))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    cid = AllChem.EmbedMolecule(mm, params)
    if cid < 0:
        raise ValueError("embed_failed")
    # Small deterministic optimization for reasonable geometry
    AllChem.UFFOptimizeMolecule(mm, maxIters=200)
    mm = Chem.RemoveHs(mm)
    return mm


def _ensure_3d(mol: Chem.Mol) -> Chem.Mol:
    if mol is None:
        raise ValueError("reference_parse_failed")
    if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
        raise ValueError("reference_no_conformer")
    return mol


def _load_reference_ligand(ref_path: Path, conformer_index: int = 0) -> Chem.Mol:
    """
    Load a reference ligand with 3D coordinates.
    Supports: .sdf, .mol, .pdb, .mol2
    For .pdbqt: convert to .pdb/.sdf/.mol2 first (recommended).
    """
    ext = ref_path.suffix.lower()

    if ext in {".sdf"}:
        suppl = Chem.SDMolSupplier(str(ref_path), removeHs=False)
        mols = [m for m in suppl if m is not None]
        if not mols:
            raise ValueError("reference_parse_failed")
        idx = 0 if conformer_index == 0 else min(conformer_index, len(mols) - 1)
        return _ensure_3d(mols[idx])

    if ext in {".mol"}:
        mol = Chem.MolFromMolFile(str(ref_path), removeHs=False, sanitize=True)
        return _ensure_3d(mol)

    if ext in {".pdb"}:
        # sanitize=False is often safer for messy PDBs; you can set True if your PDB is clean.
        mol = Chem.MolFromPDBFile(str(ref_path), removeHs=False, sanitize=False)
        if mol is None:
            raise ValueError("reference_parse_failed")
        # Optional: try sanitize after loading
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        return _ensure_3d(mol)

    if ext in {".mol2"}:
        mol = Chem.MolFromMol2File(str(ref_path), sanitize=True, removeHs=False)
        return _ensure_3d(mol)

    if ext in {".pdbqt"}:
        raise ValueError(
            "reference_format_unsupported:pdbqt "
            "(convert PDBQT to PDB/SDF/MOL2 first; RDKit doesn't reliably read PDBQT)"
        )

    raise ValueError(f"reference_format_unsupported:{ext}")


def _derive_box_from_mol(
    m: Chem.Mol, padding_A: float, min_size_A: float
) -> Dict[str, List[float]]:
    conf = m.GetConformer()
    xs, ys, zs = [], [], []
    for i in range(m.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        xs.append(float(p.x))
        ys.append(float(p.y))
        zs.append(float(p.z))

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    center = [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0]
    span_x = (max_x - min_x) + 2.0 * float(padding_A)
    span_y = (max_y - min_y) + 2.0 * float(padding_A)
    span_z = (max_z - min_z) + 2.0 * float(padding_A)

    size = [
        max(float(min_size_A), float(span_x)),
        max(float(min_size_A), float(span_y)),
        max(float(min_size_A), float(span_z)),
    ]
    return {
        "center": [float(center[0]), float(center[1]), float(center[2])],
        "size": [float(size[0]), float(size[1]), float(size[2])],
    }


def _align_local_pose_mcs(
    ligand: Chem.Mol,
    reference: Chem.Mol,
    seed: int = 0,
    timeout_s: float = 5.0,
    min_mcs_atoms: int = 3,
) -> Tuple[Chem.Mol, Dict[str, Any]]:
    """Align ligand to reference using an MCS-derived atom map.

    Returns (aligned_ligand_mol, alignment_metadata).
    """
    lig3d = _ensure_3d_conformer(ligand, seed=seed)
    ref3d = reference

    mcs = rdFMCS.FindMCS(
        [ref3d, lig3d],
        timeout=int(max(1.0, float(timeout_s))),
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        matchValences=False,
    )
    if not mcs or not mcs.smartsString:
        raise ValueError("alignment_no_mcs")
    patt = Chem.MolFromSmarts(mcs.smartsString)
    if patt is None:
        raise ValueError("alignment_no_mcs")

    ref_match = ref3d.GetSubstructMatch(patt)
    lig_match = lig3d.GetSubstructMatch(patt)
    if not ref_match or not lig_match or len(ref_match) != len(lig_match):
        raise ValueError("alignment_no_mcs")
    if len(ref_match) < int(min_mcs_atoms):
        raise ValueError("alignment_no_mcs")

    atom_map = list(zip(lig_match, ref_match))  # (probe, ref)
    rmsd = float(rdMolAlign.AlignMol(prbMol=lig3d, refMol=ref3d, atomMap=atom_map))
    meta = {"method": "mcs", "matched_atoms": int(len(atom_map)), "rmsd": float(rmsd)}
    return lig3d, meta


def _mock_energy_from_key(key: str) -> float:
    """Deterministic pseudo-energy in ~[-15, -3]."""
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    x = int(h[:12], 16)  # stable integer
    # map to [0, 12)
    y = (x % 12000) / 1000.0
    energy = -3.0 - y
    return float(energy)


class DockingRequest:
    """Engine request for a single docking/refinement evaluation.

    This keeps the cache key payload (used for hashing/equality) separate from
    potentially large engine-only payloads (e.g., an initial pose molblock).
    """

    def __init__(
        self,
        *,
        key_payload: Dict[str, Any],
        local_only: bool = False,
        initial_pose_molblock: Optional[str] = None,
        timeout_s: Optional[float] = None,
        # Engine-only payload (not included in key hashing)
        receptor_path: Optional[str] = None,
        box: Optional[Dict[str, Any]] = None,
        ligand_mol: Optional[Chem.Mol] = None,
        engine_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.key_payload = dict(key_payload)
        self.local_only = bool(local_only)
        self.initial_pose_molblock = (
            str(initial_pose_molblock) if initial_pose_molblock is not None else None
        )
        self.timeout_s = float(timeout_s) if timeout_s is not None else None
        self.receptor_path = str(receptor_path) if receptor_path is not None else None
        self.box = dict(box) if isinstance(box, dict) else None
        self.ligand_mol = ligand_mol
        self.engine_metadata = engine_metadata if engine_metadata is not None else {}


class DockingEngine:
    """Internal docking engine interface.

    Engines must be deterministic given the same inputs (or include seed in key payload).
    """

    def dock(self, request: DockingRequest) -> Tuple[float, float]:
        """Return (energy, compute_cost_units)."""
        raise NotImplementedError


class MockDockingEngine(DockingEngine):
    """Deterministic mock engine for CI/testing."""

    def dock(self, request: DockingRequest) -> Tuple[float, float]:
        # For deterministic behavior, derive energy from the cache key payload.
        # For aligned_local, also condition on initial pose if provided.
        key_payload = dict(request.key_payload)
        if request.initial_pose_molblock is not None:
            key_payload = dict(key_payload)
            key_payload["initial_pose_hash"] = _sha256_text(
                str(request.initial_pose_molblock)
            )

        key_text = _stable_json(key_payload)
        energy = _mock_energy_from_key(key_text)

        # Local-only refinement is "cheaper" than a global search in the mock engine.
        compute_cost_units = 0.25 if request.local_only else 1.0
        return float(energy), float(compute_cost_units)


class NotImplementedDockingEngine(DockingEngine):
    """Placeholder engine for unimplemented backends."""

    def __init__(self, engine_name: str) -> None:
        self.engine_name = engine_name

    def dock(self, request: DockingRequest) -> Tuple[float, float]:
        raise NotImplementedError(f"engine_not_implemented:{self.engine_name}")


class DockingEngineError(RuntimeError):
    """Engine-level failure with taxonomy-friendly reason.

    The scorer maps this to fail_reason='engine:<reason>'.
    """

    def __init__(self, reason: str, message: Optional[str] = None) -> None:
        self.reason = str(reason)
        super().__init__(message or str(reason))


class VinaFamilyCLIDockingEngine(DockingEngine):
    """Vina-family CLI backend (Vina / Smina / Gnina / QVina variants).

    This generalizes the existing vina_cli subprocess runner to other popular
    docking engines that keep a Vina-like CLI interface:

      - vina_cli  (AutoDock Vina)
      - smina_cli (Smina)
      - gnina_cli (Gnina)
      - qvina_cli (QVina / QVina2, etc.; user supplies binary_name/path)

    Reproducibility:
      - records engine path, best-effort version, sha256, full cmdline,
        bounded stdout/stderr into request.engine_metadata.
      - includes seed in the cache key payload (handled by DockingScorer).

    Ligand preparation:
      - uses optional 'meeko' to create PDBQT from RDKit mol.

    Parameters accepted via DockingScorer.params (all optional):
      - binary_path: explicit path to the docking executable
      - binary_name: name to resolve via PATH (default depends on engine)
      - version_args: list[str] (e.g., ["--version"]); best-effort
      - cpu: int
      - exhaustiveness: int
      - extra_args: list[str] additional CLI args appended verbatim
    """

    def __init__(
        self,
        *,
        engine_id: str,
        binary_name: str,
        binary_path: Optional[str] = None,
        default_timeout_s: Optional[float] = None,
        version_args: Optional[List[str]] = None,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        self.engine_id = str(engine_id)
        self.binary_name = str(binary_name)
        self.binary_path = str(binary_path) if binary_path is not None else None
        self.default_timeout_s = (
            float(default_timeout_s) if default_timeout_s is not None else None
        )
        self.version_args = list(version_args) if version_args is not None else None
        self.extra_args = list(extra_args) if extra_args is not None else None

    def _resolve_binary(self) -> str:
        if self.binary_path:
            return self.binary_path
        found = shutil.which(self.binary_name)
        return str(found) if found is not None else ""

    def _get_version(self, exe_path: str) -> Optional[str]:
        # Best-effort: engines differ in their version flags.
        # Try user-provided args first, then common fallbacks.
        arg_candidates: List[List[str]] = []
        if self.version_args:
            arg_candidates.append(list(self.version_args))
        arg_candidates += [["--version"], ["-v"]]

        for args in arg_candidates:
            try:
                out = subprocess.run(
                    [exe_path, *args],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10,
                    check=False,
                )
                txt = (out.stdout or "").strip() or (out.stderr or "").strip()
                if txt:
                    return txt.splitlines()[0].strip()
            except Exception:
                continue
        return None

    def _bounded(self, s: str, limit: int = 20000) -> str:
        s = "" if s is None else str(s)
        return s if len(s) <= limit else (s[:limit] + "\n...[truncated]...")

    def _prepare_ligand_pdbqt(self, mol: Chem.Mol) -> str:
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy  # type: ignore
        except Exception as e:
            raise DockingEngineError("parse_failed", "meeko_missing") from e

        # Meeko expects explicit Hs and 3D
        m3d = _ensure_3d_conformer(mol, seed=0)
        m3d = Chem.AddHs(Chem.Mol(m3d))

        mk_prep = MoleculePreparation()
        if hasattr(mk_prep, "prepare"):
            setups = mk_prep.prepare(m3d)  # type: ignore[attr-defined]
        else:
            setups = mk_prep(m3d)  # type: ignore[operator]

        if not setups:
            raise DockingEngineError("parse_failed", "meeko_prepare_empty")

        setup0 = setups[0]
        res = PDBQTWriterLegacy.write_string(setup0)

        # Newer Meeko returns (pdbqt_string, is_ok, err)
        if isinstance(res, tuple) and len(res) == 3:
            pdbqt_string, is_ok, err = res
            if not is_ok:
                raise DockingEngineError("parse_failed", f"meeko_write_failed:{err}")
            return str(pdbqt_string)

        # Older Meeko may return a string directly
        return str(res)

    def _parse_best_affinity(self, log_text: str) -> float:
        # Vina-family engines print a table with a first column mode index.
        # We parse the first numeric row and return the 2nd column as affinity.
        lines = [ln.strip() for ln in str(log_text).splitlines() if ln.strip()]
        for ln in lines:
            parts = ln.split()
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    return float(parts[1])
                except Exception:
                    continue
        raise DockingEngineError("parse_failed", "vina_log_no_affinity")
    
    def vina_supports_log(self, vina_path: str) -> bool:
        try:
            p = subprocess.run([vina_path, "--help"], capture_output=True, text=True)
            return "--log" in (p.stdout or "") + (p.stderr or "")
        except Exception:
            return False

    def dock(self, request: DockingRequest) -> Tuple[float, float]:
        exe_path = self._resolve_binary()
        if not exe_path:
            raise DockingEngineError("binary_missing")
        try:
            if not Path(exe_path).exists():
                raise DockingEngineError("binary_missing")
        except DockingEngineError:
            raise
        except Exception:
            raise DockingEngineError("binary_missing")

        if (
            request.receptor_path is None
            or request.box is None
            or request.ligand_mol is None
        ):
            raise DockingEngineError("parse_failed", "missing_engine_payload")

        receptor_path = str(request.receptor_path)
        center, size = _validate_box_dict(dict(request.box))

        params = dict(request.key_payload.get("params", {}) or {})
        cpu = int(params.get("cpu", 1))
        exhaustiveness = int(params.get("exhaustiveness", 8))
        seed = request.key_payload.get("seed", None)

        # Optional, verbatim CLI args appended at the end.
        extra_args: List[str] = []
        if isinstance(params.get("extra_args", None), list):
            extra_args = [str(x) for x in params.get("extra_args", [])]

        timeout_s = (
            request.timeout_s
            if request.timeout_s is not None
            else self.default_timeout_s
        )

        pdbqt_string = self._prepare_ligand_pdbqt(request.ligand_mol)

        with tempfile.TemporaryDirectory(prefix="leadopt_vinafam_") as td:
            td_p = Path(td)
            lig_p = td_p / "ligand.pdbqt"
            out_p = td_p / "out.pdbqt"
            log_p = td_p / "dock.log"

            lig_p.write_text(pdbqt_string, encoding="utf-8")

            cmd = [
                exe_path,
                "--receptor",
                receptor_path,
                "--ligand",
                str(lig_p),
                "--center_x",
                str(center[0]),
                "--center_y",
                str(center[1]),
                "--center_z",
                str(center[2]),
                "--size_x",
                str(size[0]),
                "--size_y",
                str(size[1]),
                "--size_z",
                str(size[2]),
                "--cpu",
                str(cpu),
                "--exhaustiveness",
                str(exhaustiveness),
                "--out",
                str(out_p),
                #"--log",
                #str(log_p),
            ]

            if self.vina_supports_log(exe_path):
                cmd += ["--log", str(log_p)]

            if seed is not None:
                cmd += ["--seed", str(int(seed))]

            if self.extra_args:
                cmd += [str(x) for x in self.extra_args]
            if extra_args:
                cmd += extra_args

            env = dict(os.environ)
            if "OMP_NUM_THREADS" not in env:
                env["OMP_NUM_THREADS"] = "1"

            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=float(timeout_s) if timeout_s is not None else None,
                    check=False,
                    env=env,
                )
            except subprocess.TimeoutExpired as e:
                request.engine_metadata.update(
                    {
                        "engine": self.engine_id,
                        "engine_path": exe_path,
                        "engine_version": self._get_version(exe_path),
                        "cmd": cmd,
                        "timeout_s": timeout_s,
                        "stdout": self._bounded(getattr(e, "stdout", "") or ""),
                        "stderr": self._bounded(getattr(e, "stderr", "") or ""),
                    }
                )
                raise DockingEngineError("timeout")

            stdout = self._bounded(proc.stdout or "")
            stderr = self._bounded(proc.stderr or "")

            engine_version = self._get_version(exe_path)
            engine_sha256 = None
            try:
                bp = Path(exe_path)
                if bp.exists() and bp.is_file():
                    engine_sha256 = _sha256_file(bp)
            except Exception:
                engine_sha256 = None

            request.engine_metadata.update(
                {
                    "engine": self.engine_id,
                    "engine_path": exe_path,
                    "engine_version": engine_version,
                    "engine_sha256": engine_sha256,
                    "cmd": cmd,
                    "returncode": int(proc.returncode),
                    "stdout": stdout,
                    "stderr": stderr,
                    "cpu": cpu,
                    "exhaustiveness": exhaustiveness,
                    "seed": (int(seed) if seed is not None else None),
                    "omp_num_threads": env.get("OMP_NUM_THREADS"),
                }
            )

            if proc.returncode != 0:
                raise DockingEngineError("nonzero_exit")

            try:
                log_text = (
                    log_p.read_text(encoding="utf-8") if log_p.exists() else stdout
                )
                affinity = self._parse_best_affinity(log_text)
            except DockingEngineError:
                raise
            except Exception as e:
                raise DockingEngineError("parse_failed") from e

            compute_cost_units = 0.25 if request.local_only else 1.0
            return float(affinity), float(compute_cost_units)


class VinaCLIDockingEngine(VinaFamilyCLIDockingEngine):
    """AutoDock Vina CLI backend (backwards-compatible wrapper).

    This preserves the historical class name and constructor signature used by
    DockingScorer(engine="vina_cli").
    """

    def __init__(
        self,
        *,
        binary_path: Optional[str] = None,
        default_timeout_s: Optional[float] = None,
    ) -> None:
        super().__init__(
            engine_id="vina_cli",
            binary_name="vina",
            binary_path=binary_path,
            default_timeout_s=default_timeout_s,
            version_args=["--version"],
        )

class VinaPythonDockingEngine(DockingEngine):
    """AutoDock Vina Python API backend.

    Uses the optional `vina` PyPI package (import name: `vina`).

    Expectations / constraints:
      - receptor_path is a PDBQT file (same as CLI engines)
      - ligand is converted to PDBQT using optional `meeko`
      - box is specified via center/size, used to compute maps

    This engine is best-effort across minor API differences between vina package versions.
    It records package version + parameters into request.engine_metadata for reproducibility.
    """

    def __init__(
        self,
        *,
        default_timeout_s: Optional[float] = None,
    ) -> None:
        self.default_timeout_s = (
            float(default_timeout_s) if default_timeout_s is not None else None
        )

    def _bounded(self, s: str, limit: int = 20000) -> str:
        s = "" if s is None else str(s)
        return s if len(s) <= limit else (s[:limit] + "\n...[truncated]...")

    def _prepare_ligand_pdbqt(self, mol: Chem.Mol) -> str:
        # Reuse the same Meeko-based ligand prep as the CLI engine for consistency.
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy  # type: ignore
        except Exception as e:
            raise DockingEngineError("parse_failed", "meeko_missing") from e

        m3d = _ensure_3d_conformer(mol, seed=0)
        m3d = Chem.AddHs(Chem.Mol(m3d))

        mk_prep = MoleculePreparation()
        if hasattr(mk_prep, "prepare"):
            setups = mk_prep.prepare(m3d)  # type: ignore[attr-defined]
        else:
            setups = mk_prep(m3d)  # type: ignore[operator]

        if not setups:
            raise DockingEngineError("parse_failed", "meeko_prepare_empty")

        setup0 = setups[0]
        res = PDBQTWriterLegacy.write_string(setup0)

        if isinstance(res, tuple) and len(res) == 3:
            pdbqt_string, is_ok, err = res
            if not is_ok:
                raise DockingEngineError("parse_failed", f"meeko_write_failed:{err}")
            return str(pdbqt_string)

        return str(res)

    def _vina_pkg_version(self) -> Optional[str]:
        try:
            import importlib.metadata as _ilm  # py3.8+
            return _ilm.version("vina")
        except Exception:
            return None

    def _call_with_supported_kwargs(self, fn: Any, kwargs: Dict[str, Any]) -> Any:
        try:
            import inspect

            sig = inspect.signature(fn)
            supported = {}
            for k, v in kwargs.items():
                if k in sig.parameters:
                    supported[k] = v
            return fn(**supported)
        except Exception:
            # Fallback: try calling with all kwargs; if it fails, call without kwargs.
            try:
                return fn(**kwargs)
            except Exception:
                return fn()

    def dock(self, request: DockingRequest) -> Tuple[float, float]:
        # Optional dependency
        try:
            from vina import Vina  # type: ignore
        except Exception as e:
            raise DockingEngineError("dependency_missing", "vina_missing") from e

        if (
            request.receptor_path is None
            or request.box is None
            or request.ligand_mol is None
        ):
            raise DockingEngineError("parse_failed", "missing_engine_payload")

        center, size = _validate_box_dict(dict(request.box))
        params = dict(request.key_payload.get("params", {}) or {})
        exhaustiveness = int(params.get("exhaustiveness", 8))
        n_poses = int(params.get("n_poses", 1))
        cpu = int(params.get("cpu", 1))
        sf_name = str(params.get("sf_name", "vina"))
        seed = request.key_payload.get("seed", None)

        timeout_s = (
            request.timeout_s
            if request.timeout_s is not None
            else self.default_timeout_s
        )

        ligand_pdbqt = self._prepare_ligand_pdbqt(request.ligand_mol)

        with tempfile.TemporaryDirectory(prefix="leadopt_vina_py_") as td:
            td_p = Path(td)
            lig_p = td_p / "ligand.pdbqt"
            lig_p.write_text(ligand_pdbqt, encoding="utf-8")

            engine_md: Dict[str, Any] = {
                "engine": "vina_py",
                "vina_pkg_version": self._vina_pkg_version(),
                "sf_name": sf_name,
                "cpu": cpu,
                "exhaustiveness": exhaustiveness,
                "n_poses": n_poses,
                "seed": (int(seed) if seed is not None else None),
                "timeout_s": timeout_s,
            }

            # Note: vina python API does not provide an explicit timeout mechanism.
            # We keep timeout_s in metadata for parity with CLI engines.
            try:
                v = Vina(sf_name=sf_name)  # type: ignore[call-arg]
                # receptor
                if hasattr(v, "set_receptor"):
                    v.set_receptor(str(request.receptor_path))  # type: ignore[attr-defined]
                else:
                    raise DockingEngineError("parse_failed", "vina_py_no_set_receptor")

                # maps / box
                if hasattr(v, "compute_vina_maps"):
                    self._call_with_supported_kwargs(
                        v.compute_vina_maps,  # type: ignore[attr-defined]
                        {"center": list(center), "box_size": list(size)},
                    )
                elif hasattr(v, "compute_maps"):
                    self._call_with_supported_kwargs(
                        v.compute_maps,  # type: ignore[attr-defined]
                        {"center": list(center), "box_size": list(size)},
                    )
                else:
                    raise DockingEngineError("parse_failed", "vina_py_no_compute_maps")

                # ligand
                if hasattr(v, "set_ligand_from_file"):
                    v.set_ligand_from_file(str(lig_p))  # type: ignore[attr-defined]
                else:
                    raise DockingEngineError("parse_failed", "vina_py_no_set_ligand")

                # docking
                dock_kwargs: Dict[str, Any] = {
                    "exhaustiveness": exhaustiveness,
                    "n_poses": n_poses,
                }
                # Some vina versions may accept seed / cpu-like params; pass if supported.
                if seed is not None:
                    dock_kwargs["seed"] = int(seed)
                if cpu is not None:
                    dock_kwargs["cpu"] = int(cpu)
                    dock_kwargs["n_cpu"] = int(cpu)
                    dock_kwargs["num_cpus"] = int(cpu)

                if hasattr(v, "dock"):
                    self._call_with_supported_kwargs(
                        v.dock,  # type: ignore[attr-defined]
                        dock_kwargs,
                    )
                else:
                    raise DockingEngineError("parse_failed", "vina_py_no_dock")

                # energies
                affinity: Optional[float] = None
                if hasattr(v, "energies"):
                    try:
                        e = v.energies(n_poses=1)  # type: ignore[attr-defined]
                        # expected shape: [[affinity, ...], ...] or [affinity, ...]
                        if isinstance(e, list) and e:
                            first = e[0]
                            if isinstance(first, (list, tuple)) and first:
                                affinity = float(first[0])
                            else:
                                affinity = float(first)
                    except Exception:
                        affinity = None
                if affinity is None and hasattr(v, "score"):
                    try:
                        s = v.score()  # type: ignore[attr-defined]
                        if isinstance(s, (list, tuple)) and s:
                            affinity = float(s[0])
                        else:
                            affinity = float(s)
                    except Exception:
                        affinity = None

                if affinity is None:
                    raise DockingEngineError("parse_failed", "vina_py_no_affinity")

            except DockingEngineError:
                request.engine_metadata.update(engine_md)
                raise
            except Exception as e:
                engine_md["exception_type"] = type(e).__name__
                engine_md["exception_message"] = str(e)
                request.engine_metadata.update(engine_md)
                raise DockingEngineError("engine_failure", "vina_py_exception") from e

            request.engine_metadata.update(engine_md)
            compute_cost_units = 0.25 if request.local_only else 1.0
            return float(affinity), float(compute_cost_units)

@dataclass
class DockingScorer(Scorer):
    """
    DockingScorer skeleton.

    Phase 4.2 provides:
      - API-compliant scorer with deterministic mock engine for testing
      - standardized objective direction: objective = -energy
      - failure-safe behavior
      - metadata logging: receptor hash, box, params, compute_cost

    Real docking engines and caching arrive in Phase 4.3+.
    """

    engine: str
    protocol: str
    receptor_path: str

    # Behavioral implementation (alignment + local-only request) is added in later stages.
    reference_ligand_path: Optional[str] = None
    # Backwards-compatible alias for early adopters; prefer reference_ligand_path going forward.
    reference_ligand: Optional[str] = None
    reference_conformer: Optional[int] = None
    alignment: Optional[Dict[str, Any]] = None
    anchor: Optional[Dict[str, Any]] = None
    local_opt: Optional[Dict[str, Any]] = None

    # Exactly one of these should be provided. Validation is handled by preset_yaml,
    # but we keep runtime guardrails too.
    box: Optional[Dict[str, Any]] = None
    box_file: Optional[str] = None

    # Engine-specific knobs (stored and included in mock key)
    params: Optional[Dict[str, Any]] = None

    # Reproducibility / controls
    seed: Optional[int] = None
    timeout_s: Optional[float] = None
    budget: Optional[int] = None

    # Cache is Phase 4.3; we keep the field for forward compatibility
    cache_dir: Optional[str] = None

    # Internal runtime state (per-scorer-instance)
    _calls_made: int = 0  # number of cache-miss docking computations performed
    _cache_hits: int = 0
    _cache: Optional[_DockingCache] = None

    def __post_init__(self) -> None:
        if self.cache_dir:
            self._cache = _DockingCache(self.cache_dir)

    # Engine/version strings for cache keys and manifests
    engine_version: str = "0"
    version_override: str = "0"

    # Failure behavior
    fail_objective: float = Scorer.fail_objective

    @property
    def version(self) -> str:
        return str(self.version_override)

    def _resolve_reference_ligand_path(self) -> Optional[Path]:
        p = self.reference_ligand_path or self.reference_ligand
        if p is None:
            return None
        return Path(str(p))

    def scorer_metadata(self) -> Dict[str, Any]:
        # Do not compute receptor hash here (may access filesystem); do it in score().
        return {
            "name": self.name,
            "version": self.version,
            "type": "docking",
            "engine": str(self.engine),
            "engine_version": str(self.engine_version),
            "protocol": str(self.protocol),
            "receptor_path": str(self.receptor_path),
            "reference_ligand_path": (
                str(self.reference_ligand_path)
                if self.reference_ligand_path is not None
                else (
                    str(self.reference_ligand)
                    if self.reference_ligand is not None
                    else None
                )
            ),
            "reference_conformer": (
                int(self.reference_conformer)
                if self.reference_conformer is not None
                else None
            ),
            "alignment": (
                dict(self.alignment) if isinstance(self.alignment, dict) else None
            ),
            "anchor": dict(self.anchor) if isinstance(self.anchor, dict) else None,
            "local_opt": (
                dict(self.local_opt) if isinstance(self.local_opt, dict) else None
            ),
            "seed": int(self.seed) if self.seed is not None else None,
            "timeout_s": float(self.timeout_s) if self.timeout_s is not None else None,
            "budget": int(self.budget) if self.budget is not None else None,
            "cache_dir": str(self.cache_dir) if self.cache_dir is not None else None,
            # box/box_file are included in result metadata (score-time), because box may be normalized
        }

    def _make_engine(self) -> DockingEngine:
        engine = str(self.engine).strip().lower()

        # ------------------------------------------------------------------
        # Mock engine (used for CI / dev)
        # ------------------------------------------------------------------
        if engine == "mock":
            return MockDockingEngine()

        # ------------------------------------------------------------------
        # AutoDock Vina CLI (original implementation, backward compatible)
        # ------------------------------------------------------------------
        if engine == "vina_cli":
            binary_path = None
            if isinstance(self.params, dict):
                binary_path = self.params.get("binary_path", None)

            return VinaCLIDockingEngine(
                binary_path=binary_path,
                default_timeout_s=self.timeout_s,
            )

        # ------------------------------------------------------------------
        # AutoDock Vina Python API backend
        # Requires: pip install vina
        # ------------------------------------------------------------------
        if engine in {"vina_py", "vina_python"}:
            return VinaPythonDockingEngine(
                default_timeout_s=self.timeout_s,
            )

        # ------------------------------------------------------------------
        # Vina-family CLI engines (Smina / Gnina / QVina variants)
        # ------------------------------------------------------------------
        if engine in {"smina_cli", "gnina_cli", "qvina_cli"}:
            params = dict(self.params) if isinstance(self.params, dict) else {}

            binary_path = params.get("binary_path", None)
            binary_name = params.get("binary_name", None)

            # Reasonable defaults; user can override via params.binary_name/binary_path.
            default_name = {
                "smina_cli": "smina",
                "gnina_cli": "gnina",
                "qvina_cli": "qvina2",
            }[engine]

            version_args = None
            if isinstance(params.get("version_args", None), list):
                version_args = [str(x) for x in params.get("version_args", [])]

            return VinaFamilyCLIDockingEngine(
                engine_id=engine,
                binary_name=str(binary_name) if binary_name is not None else default_name,
                binary_path=str(binary_path) if binary_path is not None else None,
                default_timeout_s=self.timeout_s,
                version_args=version_args,
            )

        # ------------------------------------------------------------------
        # Unsupported engine
        # ------------------------------------------------------------------
        return NotImplementedDockingEngine(engine_name=str(self.engine))

    def _key_payload_and_hash(
        self,
        canonical_smiles: str,
        receptor_hash: str,
        box_repr: Any,
        protocol_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        payload: Dict[str, Any] = {
            "cache_schema_version": int(_DOCKING_CACHE_SCHEMA_VERSION),
            "scorer_version": str(self.version),
            "smiles": canonical_smiles,
            "receptor_hash": receptor_hash,
            "protocol": str(self.protocol),
            "engine": str(self.engine),
            "engine_version": str(self.engine_version),
            "box": box_repr,
            "params": dict(self.params) if self.params else {},
            # include seed to avoid silently mixing stochastic engines later
            "seed": int(self.seed) if self.seed is not None else None,
        }

        if protocol_payload:
            # Protocol-specific payload fields must be included in the key to ensure
            # cache correctness across protocol variants.
            payload["protocol_payload"] = dict(protocol_payload)
        key_text = _stable_json(payload)
        return payload, _sha256_text(key_text)

    def score(
        self, mol: Any, context: Optional[Dict[str, Any]] = None
    ) -> ScoringResult:
        t0 = time.perf_counter()
        try:
            if mol is None:
                return ScoringResult(
                    objective=float(self.fail_objective),
                    components={"objective": float(self.fail_objective)},
                    valid=False,
                    fail_reason="input:mol_is_none",
                    metadata={**self.scorer_metadata()},
                )

            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            if not canonical_smiles:
                return ScoringResult(
                    objective=float(self.fail_objective),
                    components={"objective": float(self.fail_objective)},
                    valid=False,
                    fail_reason="input:empty_smiles",
                    metadata={**self.scorer_metadata()},
                )

            receptor_p = Path(self.receptor_path)
            if not receptor_p.exists():
                # Also allow presets to reference packaged data paths.
                rel = str(self.receptor_path).replace("\\", "/").lstrip("/")
                if rel.startswith("leadopt/"):
                    rel = rel[len("leadopt/") :]
                candidate = files("leadopt").joinpath(rel)
                receptor_p = Path(str(candidate))

            if not receptor_p.exists():
                md = {**self.scorer_metadata(), "smiles": canonical_smiles}
                md["compute_cost_s"] = float(time.perf_counter() - t0)
                md["receptor_path_resolved"] = None
                return ScoringResult(
                    objective=float(self.fail_objective),
                    components={
                        "objective": float(self.fail_objective),
                        "compute_cost_s": md["compute_cost_s"],
                    },
                    valid=False,
                    fail_reason="input:receptor_not_found",
                    metadata=md,
                )

            receptor_hash = _sha256_file(receptor_p)

            # box representation (normalized)
            box_repr: Any
            if self.box is not None:
                center, size = _validate_box_dict(self.box)
                box_repr = {"center": list(center), "size": list(size)}
            elif self.box_file is not None:
                box_repr = {"box_file": str(self.box_file)}
            else:
                # Should not happen if preset_yaml validation is used
                raise ValueError("missing_box_and_box_file")

            protocol = str(self.protocol).strip().lower()
            if protocol not in {"standard", "aligned_local"}:
                raise ValueError(f"unknown_protocol:{self.protocol}")

            initial_pose_molblock: Optional[str] = None
            protocol_payload: Dict[str, Any] = {}
            if protocol == "aligned_local":
                ref_p = self._resolve_reference_ligand_path()
                if ref_p is None:
                    md = {**self.scorer_metadata(), "smiles": canonical_smiles}
                    md["compute_cost_s"] = float(time.perf_counter() - t0)
                    return ScoringResult(
                        objective=float(self.fail_objective),
                        components={
                            "objective": float(self.fail_objective),
                            "compute_cost_s": md["compute_cost_s"],
                        },
                        valid=False,
                        fail_reason="input:missing_reference_ligand",
                        metadata=md,
                    )
                if not ref_p.exists():
                    md = {**self.scorer_metadata(), "smiles": canonical_smiles}
                    md["compute_cost_s"] = float(time.perf_counter() - t0)
                    md["reference_ligand_path"] = str(ref_p)
                    return ScoringResult(
                        objective=float(self.fail_objective),
                        components={
                            "objective": float(self.fail_objective),
                            "compute_cost_s": md["compute_cost_s"],
                        },
                        valid=False,
                        fail_reason="input:missing_reference_ligand",
                        metadata=md,
                    )

                reference_ligand_hash = _sha256_file(ref_p)
                protocol_payload["reference_ligand_hash"] = str(reference_ligand_hash)
                protocol_payload["reference_conformer"] = (
                    int(self.reference_conformer)
                    if self.reference_conformer is not None
                    else None
                )
                protocol_payload["alignment"] = (
                    dict(self.alignment) if isinstance(self.alignment, dict) else None
                )
                protocol_payload["anchor"] = (
                    dict(self.anchor) if isinstance(self.anchor, dict) else None
                )
                protocol_payload["local_opt"] = (
                    dict(self.local_opt) if isinstance(self.local_opt, dict) else None
                )

                # Stage 3 (Phase 6.1): deterministic RDKit alignment + derived local box.
                try:
                    ref_mol = _load_reference_ligand(
                        ref_p,
                        conformer_index=(
                            int(self.reference_conformer)
                            if self.reference_conformer is not None
                            else 0
                        ),
                    )
                except ValueError as ve:
                    md = {**self.scorer_metadata(), "smiles": canonical_smiles}
                    md["compute_cost_s"] = float(time.perf_counter() - t0)
                    md["reference_ligand_path"] = str(ref_p)
                    reason = str(ve)
                    if reason == "reference_no_conformer":
                        fail_reason = "input:reference_no_conformer"
                    else:
                        fail_reason = "input:reference_parse_failed"
                    return ScoringResult(
                        objective=float(self.fail_objective),
                        components={
                            "objective": float(self.fail_objective),
                            "compute_cost_s": md["compute_cost_s"],
                        },
                        valid=False,
                        fail_reason=fail_reason,
                        metadata=md,
                    )

                align_cfg = (
                    dict(self.alignment) if isinstance(self.alignment, dict) else {}
                )
                align_method = str(align_cfg.get("method", "mcs")).strip().lower()
                align_timeout_s = float(align_cfg.get("timeout_s", 5.0))
                min_mcs_atoms = int(align_cfg.get("min_mcs_atoms", 3))

                try:
                    if align_method != "mcs":
                        # Stage 3 implements MCS alignment only; other methods reserved for later stages.
                        raise ValueError("alignment_method_not_supported")
                    aligned_mol, align_meta = _align_local_pose_mcs(
                        ligand=mol,
                        reference=ref_mol,
                        seed=int(self.seed) if self.seed is not None else 0,
                        timeout_s=float(align_timeout_s),
                        min_mcs_atoms=int(min_mcs_atoms),
                    )
                except ValueError as ve:
                    md = {**self.scorer_metadata(), "smiles": canonical_smiles}
                    md["compute_cost_s"] = float(time.perf_counter() - t0)
                    md["reference_ligand_path"] = str(ref_p)
                    reason = str(ve)
                    if reason in {"alignment_no_mcs", "embed_failed"}:
                        fail_reason = "input:alignment_no_mcs"
                    elif reason == "alignment_method_not_supported":
                        fail_reason = "input:alignment_method_not_supported"
                    else:
                        fail_reason = f"exception:{type(ve).__name__}"
                    return ScoringResult(
                        objective=float(self.fail_objective),
                        components={
                            "objective": float(self.fail_objective),
                            "compute_cost_s": md["compute_cost_s"],
                        },
                        valid=False,
                        fail_reason=fail_reason,
                        metadata=md,
                    )

                initial_pose_molblock = Chem.MolToMolBlock(aligned_mol)
                protocol_payload["_initial_pose_hash"] = _sha256_text(
                    initial_pose_molblock
                )

                anchor_cfg = dict(self.anchor) if isinstance(self.anchor, dict) else {}
                padding_A = float(anchor_cfg.get("box_padding_A", 1.5))
                min_size_A = float(anchor_cfg.get("box_min_A", 6.0))
                derive_box = bool(anchor_cfg.get("derive_box", True))

                derived_box = _derive_box_from_mol(
                    aligned_mol, padding_A=padding_A, min_size_A=min_size_A
                )
                if derive_box:
                    box_repr = dict(derived_box)

                # Attach alignment/box details (includes derived box) for transparency & cache correctness.
                protocol_payload["_alignment_meta"] = {
                    "method": str(align_meta.get("method", "mcs")),
                    "matched_atoms": int(align_meta.get("matched_atoms", 0)),
                }
                protocol_payload["_derived_box"] = dict(derived_box)

            engine_obj = self._make_engine()

            # Cache key (spec: smiles + receptor_hash + protocol + box + params + engine/version + seed)
            key_payload, key_sha256 = self._key_payload_and_hash(
                canonical_smiles,
                receptor_hash,
                box_repr,
                protocol_payload=(protocol_payload if protocol_payload else None),
            )

            # Cache lookup (if enabled)
            cached_energy: Optional[float] = None
            cache_hit = False
            if self._cache is not None:
                entry = self._cache.get(key_sha256)
                if entry is not None:
                    # Strong validity check: only accept cache hits if the stored payload
                    # exactly matches the current payload. This protects against schema
                    # drift, corruption, and any theoretical collisions.
                    if dict(entry.key_payload) == dict(key_payload):
                        cached_energy = float(entry.docking_energy)
                        cache_hit = True
                        self._cache_hits += 1
                    else:
                        cache_hit = False
                        cached_energy = None

            # Budget enforcement applies only to cache misses (real docking computations)
            if (not cache_hit) and (self.budget is not None):
                if self._calls_made >= int(self.budget):
                    compute_cost_s = float(time.perf_counter() - t0)
                    md = {
                        **self.scorer_metadata(),
                        "smiles": canonical_smiles,
                        "receptor_hash": receptor_hash,
                        "box": box_repr,
                        "cache_hit": False,
                        "compute_cost_s": float(compute_cost_s),
                        "compute_cost_units": 0.0,
                        "compute_cost": 0.0,
                        "calls_made": int(self._calls_made),
                        "cache_hits": int(self._cache_hits),
                        "key_sha256": key_sha256,
                    }
                    if context:
                        md["context"] = dict(context)
                    return ScoringResult(
                        objective=float(self.fail_objective),
                        components={
                            "objective": float(self.fail_objective),
                            "compute_cost_s": float(compute_cost_s),
                        },
                        valid=False,
                        fail_reason="budget:exceeded",
                        metadata=md,
                    )

            # Deterministic mock docking (cache miss -> compute)
            if cache_hit:
                energy = float(cached_energy)
                compute_cost_units = 0.0
            else:
                # Cache miss => perform docking computation via engine adapter.
                engine_metadata: Dict[str, Any] = {}
                request = DockingRequest(
                    key_payload=key_payload,
                    local_only=(protocol == "aligned_local"),
                    initial_pose_molblock=(
                        initial_pose_molblock if protocol == "aligned_local" else None
                    ),
                    timeout_s=self.timeout_s,
                    receptor_path=str(receptor_p),
                    box=(box_repr if isinstance(box_repr, dict) else None),
                    ligand_mol=mol,
                    engine_metadata=engine_metadata,
                )
                try:
                    energy, compute_cost_units = engine_obj.dock(request)
                except DockingEngineError as e:
                    compute_cost_s = float(time.perf_counter() - t0)
                    md = {
                        **self.scorer_metadata(),
                        "smiles": canonical_smiles,
                        "receptor_hash": receptor_hash,
                        "box": box_repr,
                        "cache_hit": False,
                        "compute_cost_s": float(compute_cost_s),
                        "compute_cost_units": 0.0,
                        "compute_cost": 0.0,
                        "calls_made": int(self._calls_made),
                        "cache_hits": int(self._cache_hits),
                        "key_sha256": key_sha256,
                        "engine_provenance": dict(engine_metadata),
                    }
                    if context:
                        md["context"] = dict(context)
                    return ScoringResult(
                        objective=float(self.fail_objective),
                        components={
                            "objective": float(self.fail_objective),
                            "compute_cost_s": float(compute_cost_s),
                        },
                        valid=False,
                        fail_reason=f"engine:{e.reason}",
                        metadata=md,
                    )
                except NotImplementedError as e:
                    compute_cost_s = float(time.perf_counter() - t0)
                    md = {
                        **self.scorer_metadata(),
                        "smiles": canonical_smiles,
                        "receptor_hash": receptor_hash,
                        "box": box_repr,
                        "cache_hit": False,
                        "compute_cost_s": float(compute_cost_s),
                        "compute_cost_units": 0.0,
                        "calls_made": int(self._calls_made),
                        "cache_hits": int(self._cache_hits),
                        "key_sha256": key_sha256,
                    }
                    if context:
                        md["context"] = dict(context)

                    reason = str(e)
                    if reason.startswith("engine_not_implemented:"):
                        reason = (
                            "engine:not_implemented:"
                            + reason.split("engine_not_implemented:", 1)[1]
                        )
                    else:
                        reason = "engine:" + reason
                    return ScoringResult(
                        objective=float(self.fail_objective),
                        components={
                            "objective": float(self.fail_objective),
                            "compute_cost_s": float(compute_cost_s),
                        },
                        valid=False,
                        fail_reason=reason,
                        metadata=md,
                    )

                energy = float(energy)
                compute_cost_units = float(compute_cost_units)
                self._calls_made += 1

                # Write cache on miss
                if self._cache is not None:
                    self._cache.set(
                        _DockingCacheEntry(
                            key_sha256=key_sha256,
                            key_payload=key_payload,
                            docking_energy=float(energy),
                            engine=str(self.engine).strip().lower(),
                            engine_version=str(self.engine_version),
                            protocol=str(self.protocol).strip().lower(),
                            receptor_hash=receptor_hash,
                            box=box_repr,
                            params=dict(self.params) if self.params else {},
                            seed=(int(self.seed) if self.seed is not None else None),
                        )
                    )

            objective = -float(energy)  # standardized direction: higher is better
            compute_cost_s = float(time.perf_counter() - t0)

            components: Dict[str, float] = {
                "docking_energy": float(energy),
                "objective": float(objective),
                "compute_cost_s": float(compute_cost_s),
                "compute_cost_units": float(compute_cost_units),
                # Back-compat alias (RewardComposer historically used metadata["compute_cost"])
                "compute_cost": float(compute_cost_units),
            }

            md = {
                **self.scorer_metadata(),
                "smiles": canonical_smiles,
                "receptor_hash": receptor_hash,
                "box": box_repr,
                **(
                    {"protocol_payload": dict(protocol_payload)}
                    if protocol_payload
                    else {}
                ),
                "cache_hit": bool(cache_hit),
                "compute_cost_s": float(compute_cost_s),
                "compute_cost_units": float(compute_cost_units),
                # Back-compat alias
                "compute_cost": float(compute_cost_units),
                "calls_made": int(self._calls_made),
                "cache_hits": int(self._cache_hits),
                "key_sha256": key_sha256,
                "engine_provenance": (
                    dict(engine_metadata) if "engine_metadata" in locals() else None
                ),
            }

            if context:
                md["context"] = dict(context)

            return ScoringResult(
                objective=float(objective),
                components=components,
                valid=True,
                fail_reason=None,
                metadata=md,
            )

        except Exception as e:
            compute_cost_s = float(time.perf_counter() - t0)
            md = {**self.scorer_metadata()}
            md["exception_type"] = type(e).__name__
            md["exception_message"] = str(e)
            md["compute_cost_s"] = float(compute_cost_s)
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

            return ScoringResult(
                objective=float(self.fail_objective),
                components={
                    "objective": float(self.fail_objective),
                    "compute_cost_s": float(compute_cost_s),
                },
                valid=False,
                fail_reason=f"exception:{type(e).__name__}",
                metadata=md,
            )
