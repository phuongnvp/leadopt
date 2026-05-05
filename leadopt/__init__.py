"""leadopt package.

Academic reproducibility note:
  - __version__ is derived from installed package metadata when available.
  - Source checkouts without installation fall back to a conservative placeholder.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _get_version() -> str:
    try:
        return version("leadopt")
    except PackageNotFoundError:
        # Source checkout / not installed.
        return "0.0.0+unknown"


__version__ = _get_version()

__all__ = [
    "__version__",
    "core",
    "constraints",
    "actions",
    "env",
    "models",
    "rl",
    "sar",
    "scoring",
    "api",
]
