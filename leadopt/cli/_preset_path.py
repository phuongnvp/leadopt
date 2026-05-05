from __future__ import annotations

from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterator, Union


@contextmanager
def resolve_preset_path(preset: Union[str, Path]) -> Iterator[Path]:
    """Resolve a preset path for CLI usage.

    Accepts either:
      - a real filesystem path
      - a package-relative path like "presets/foo.yaml"
      - a repo-ish path like "leadopt/presets/foo.yaml"

    Returns a concrete Path usable with Path(...).read_text().
    """

    p = Path(str(preset))
    if p.exists():
        yield p
        return

    s = str(preset).replace("\\", "/").lstrip("/")
    if s.startswith("leadopt/"):
        s = s[len("leadopt/") :]
    if not s.startswith("presets/"):
        # Common user convenience:
        #   - allow "medchem_quality_tier4" (no extension)
        #   - allow "medchem_quality_tier4.yaml" (filename)
        if "/" not in s:
            if s.endswith(".yaml"):
                s = f"presets/{s}"
            else:
                s = f"presets/{s}.yaml"

    candidate = files("leadopt").joinpath(s)
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Preset not found. Tried filesystem path '{preset}' and package resource '{s}'."
        )

    with as_file(candidate) as fp:
        yield Path(fp)
