from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def resolve_preset_path(preset: Union[str, Path, None]) -> Optional[Path]:
    """Resolve a preset name or path into an existing YAML file path.

    Design constraint:
    - CLI remains source-of-truth for how presets are located.
    - This API helper reuses the CLI resolver.

    Args:
        preset: Either a preset name (e.g. "medchem_quality_tier4"),
            a filesystem path to a YAML preset, or None.

    Returns:
        Path to an existing preset YAML, or None if preset is None.
    """
    if preset is None:
        return None

    # Keep imports light at module import time.
    from leadopt.cli._preset_path import resolve_preset_path as _cli_resolve_preset_path

    with _cli_resolve_preset_path(str(preset)) as preset_path:
        return Path(preset_path)


__all__ = ["resolve_preset_path"]
