# leadopt/core/seeding.py
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SeedReport:
    seed: int
    deterministic_torch: bool
    cudnn_deterministic: Optional[bool]
    cudnn_benchmark: Optional[bool]
    cublas_workspace_config: Optional[str]


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> SeedReport:
    """
    Set global RNG seeds for reproducible experiments.

    This framework already provides a deterministic *MDP* (action enumeration/masking).
    This function makes *experiments* more reproducible by aligning:
      - Python's `random`
      - NumPy RNG
      - PyTorch RNG (CPU and CUDA, if available)
      - Optional: enforce deterministic PyTorch algorithms where possible

    Notes:
      - Full bit-for-bit determinism on GPU can still be hardware/ops dependent.
      - For strict CUDA determinism, you may also need:
          export CUBLAS_WORKSPACE_CONFIG=:4096:8
        before launching Python.
      - We do not set PYTHONHASHSEED here (must be set before interpreter start).
    """
    seed = int(seed)

    random.seed(seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    cudnn_det = None
    cudnn_bench = None
    cublas_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            # Enforce deterministic algorithms where supported.
            # If a nondeterministic op is used, this can raise an error (good for debugging).
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # Older torch versions may not support this; ignore gracefully.
                pass

            # cuDNN flags (relevant for CNNs; safe to set anyway)
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                cudnn_det = True
                cudnn_bench = False
            except Exception:
                pass

    except Exception:
        # torch not installed or unavailable: ignore
        pass

    return SeedReport(
        seed=seed,
        deterministic_torch=bool(deterministic_torch),
        cudnn_deterministic=cudnn_det,
        cudnn_benchmark=cudnn_bench,
        cublas_workspace_config=cublas_cfg,
    )
