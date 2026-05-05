from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    lo: float
    hi: float
    n: int


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    seed: int = 0,
) -> BootstrapCI:
    """
    Nonparametric bootstrap CI for a statistic (default mean).

    Returns (mean, lo, hi, n). Deterministic given seed.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return BootstrapCI(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    if n == 1:
        v = float(arr[0])
        return BootstrapCI(mean=v, lo=v, hi=v, n=1)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = arr[idx]
    stats = np.apply_along_axis(stat_fn, 1, samples)

    point = float(stat_fn(arr))
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return BootstrapCI(mean=point, lo=lo, hi=hi, n=n)


def format_ci(ci: BootstrapCI, digits: int = 3) -> str:
    if ci.n == 0 or np.isnan(ci.mean):
        return "NA"
    return f"{ci.mean:.{digits}f} [{ci.lo:.{digits}f}, {ci.hi:.{digits}f}] (n={ci.n})"
