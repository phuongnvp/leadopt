from __future__ import annotations

"""leadopt.core public API.

This module is intentionally import-light so that the *base* installation of leadopt
(without optional heavy chemistry backends) can still import and expose helpful
error messages.

RDKit-backed utilities are available when RDKit is installed (see extras: leadopt[chem]).
"""

from typing import Any, TypeVar

from .errors import ActionError
from .seeding import SeedReport, set_global_seed

T = TypeVar("T")


def _require_rdkit() -> None:
    try:
        import rdkit  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "RDKit-backed core utilities are not available because RDKit is not installed. "
            "Install with 'pip install leadopt[chem]' or via conda-forge "
            "('conda install -c conda-forge rdkit')."
        ) from e


# --- RDKit-backed symbols (lazy stubs when RDKit is missing) -----------------

try:
    from .mol import MoleculeState as MoleculeState  # noqa: F401
    from .rdkit_utils import assert_valid_mol as assert_valid_mol  # noqa: F401
    from .rdkit_utils import canonical_smiles as canonical_smiles  # noqa: F401
    from .rules import RuleConfig as RuleConfig  # noqa: F401
    from .rules import check_molecule as check_molecule  # noqa: F401
except ImportError:

    class MoleculeState:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_rdkit()

    def canonical_smiles(*args: Any, **kwargs: Any) -> str:  # type: ignore[no-redef]
        _require_rdkit()
        raise AssertionError("unreachable")

    def assert_valid_mol(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        _require_rdkit()
        raise AssertionError("unreachable")

    class RuleConfig:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_rdkit()

    def check_molecule(*args: Any, **kwargs: Any) -> bool:  # type: ignore[no-redef]
        _require_rdkit()
        raise AssertionError("unreachable")


__all__ = [
    "ActionError",
    "MoleculeState",
    "canonical_smiles",
    "assert_valid_mol",
    "RuleConfig",
    "check_molecule",
    "set_global_seed",
    "SeedReport",
]
