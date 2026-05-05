# leadopt/core/run_logging.py
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    # Python 3.8+
    from importlib.metadata import version as pkg_version
except Exception:  # pragma: no cover
    pkg_version = None  # type: ignore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_commit_hash(repo_root: Optional[Path] = None) -> str:
    """
    Return git commit hash if available, else "unknown".
    Works even if git is missing or code is not in a repo.
    """
    cwd = str(repo_root) if repo_root is not None else None
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _safe_json(obj: Any) -> Any:
    """
    Convert objects into JSON-serializable structures.
    """
    if obj is None:
        return None

    # Basic JSON types
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_safe_json(x) for x in obj]
    if isinstance(obj, tuple):
        return [_safe_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}

    # dataclasses
    if is_dataclass(obj):
        return _safe_json(asdict(obj))

    # Path
    if isinstance(obj, Path):
        return str(obj)

    # Fallback
    return repr(obj)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(_safe_json(payload), indent=2, sort_keys=True))


def resolve_repo_root(start: Optional[Path] = None) -> Optional[Path]:
    """
    Best-effort repo root resolution:
    - walk upward looking for .git/
    """
    p = (start or Path.cwd()).resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def make_run_dir(base_dir: Path, run_name: Optional[str] = None) -> Path:
    """
    Create a new run directory.
    Default name: YYYYmmdd_HHMMSS (UTC) or with suffix run_name.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = ts if not run_name else f"{ts}_{run_name}"
    run_dir = base_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_preset_used(
    run_dir: Path, preset_source_path: Optional[Path], preset_raw: Dict[str, Any]
) -> Path:
    """
    Save the exact YAML used if source path exists; otherwise dump a canonical YAML-ish JSON.
    Phase 4 will ensure we always have a source file in normal usage.
    """
    out_path = run_dir / "preset_used.yaml"

    if preset_source_path is not None and preset_source_path.exists():
        write_bytes_atomic(out_path, preset_source_path.read_bytes())
        return out_path

    # Fallback: store JSON with .yaml extension (still reproducible, but not byte-identical)
    write_text_atomic(
        out_path, json.dumps(_safe_json(preset_raw), indent=2, sort_keys=True)
    )
    return out_path


def build_run_manifest(
    *,
    preset_name: str,
    preset_path: Optional[Path],
    seed: Optional[int],
    scorer: Any,
    constraint_suite: Any,
    reward_composer: Any,
    env_kwargs: Dict[str, Any],
    training_meta: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    repo_root = resolve_repo_root(preset_path.parent if preset_path else None)
    git_hash = get_git_commit_hash(repo_root)

    leadopt_ver = "unknown"
    if pkg_version is not None:
        try:
            leadopt_ver = pkg_version("leadopt")
        except Exception:
            leadopt_ver = "unknown"

    scorer_meta = None
    if scorer is not None and hasattr(scorer, "scorer_metadata"):
        try:
            scorer_meta = scorer.scorer_metadata()
        except Exception:
            scorer_meta = {"error": "scorer_metadata() failed"}

    constraint_meta = None
    if constraint_suite is not None and hasattr(constraint_suite, "metadata"):
        try:
            constraint_meta = constraint_suite.metadata()
        except Exception:
            constraint_meta = {"error": "constraint_suite.metadata() failed"}

    manifest: Dict[str, Any] = {
        "created_utc": _utc_now_iso(),
        "preset": {
            "name": preset_name,
            "path": str(preset_path) if preset_path else None,
        },
        "reproducibility": {
            "seed": seed,
            "git_commit": git_hash,
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "env": {
                # keep minimal and non-sensitive; add more if you want
                "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", None),
            },
            "leadopt_version": leadopt_ver,
        },
        "scoring": {
            "scorer_class": scorer.__class__.__name__ if scorer is not None else None,
            "scorer_metadata": scorer_meta,
        },
        "constraints": {
            "enabled": constraint_suite is not None,
            "metadata": constraint_meta,
        },
        "reward": {
            "reward_composer_class": (
                reward_composer.__class__.__name__
                if reward_composer is not None
                else None
            ),
            "reward_composer": reward_composer,
        },
        "environment": {
            "env_kwargs": env_kwargs,
        },
        "training": {
            "meta": training_meta,
        },
    }

    if extra:
        manifest["extra"] = extra

    return manifest


def write_run_artifacts(
    *,
    run_dir: Path,
    preset_source_path: Optional[Path],
    preset_raw: Dict[str, Any],
    preset_name: str,
    seed: Optional[int],
    scorer: Any,
    constraint_suite: Any,
    reward_composer: Any,
    env_kwargs: Dict[str, Any],
    training_meta: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Writes:
      - preset_used.yaml
      - run_manifest.json
    Returns paths.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    preset_path = save_preset_used(run_dir, preset_source_path, preset_raw)

    manifest = build_run_manifest(
        preset_name=preset_name,
        preset_path=preset_source_path,
        seed=seed,
        scorer=scorer,
        constraint_suite=constraint_suite,
        reward_composer=reward_composer,
        env_kwargs=env_kwargs,
        training_meta=training_meta,
        extra=extra,
    )
    manifest_path = run_dir / "run_manifest.json"
    write_json(manifest_path, manifest)

    return {"preset_used": preset_path, "manifest": manifest_path}
