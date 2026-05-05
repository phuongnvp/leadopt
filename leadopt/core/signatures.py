from __future__ import annotations

"""Small shared helpers for reproducibility.

These helpers are intentionally dependency-light so they can be imported in the
base installation (without RDKit/Torch).
"""

import hashlib
import json
from typing import Any, Iterable


def operator_signature(operators: Iterable[Any]) -> str:
    """Return a stable signature for an ordered sequence of action operators.

    The signature is computed from the operator class names and their ``repr``.

    Notes
    -----
    - This is intended for *reproducibility bookkeeping* (detecting action-space
      drift between training and generation).
    - If an operator's ``repr`` changes across versions, the signature will
      change as well.
    """

    payload = [{"cls": op.__class__.__name__, "repr": repr(op)} for op in operators]
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()