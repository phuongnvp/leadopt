#!/usr/bin/env python3
"""
wrap_model.py

Wrap an existing pickled QSAR regression model (predicting raw pIC50) so that
model.predict(...) returns a bounded reward in (0,1) using a logistic transform:

    reward = sigmoid((pIC50 - center) / slope)

Usage:
  python wrap_model.py \
      --in_model /path/to/base_model.pkl \
      --out_model /path/to/wrapped_model.pkl \
      --center 7.0 \
      --slope 1.0
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    # For large negative z, exp(z) underflows safely; for large positive z, exp(-z) underflows safely.
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class LogisticSquashedRegressor:
    """
    A minimal wrapper that preserves the predict(X) API.

    It returns reward in (0,1) computed from the base model's raw prediction (pIC50).
    """
    base_model: Any
    center: float = 7.0
    slope: float = 1.0
    clip_z: float = 30.0  # Avoid extreme exponent overflow; sigmoid(±30) is effectively saturated.

    def predict(self, X: Any) -> np.ndarray:
        """
        Parameters
        ----------
        X
            Whatever the base model accepts (typically numpy array of features, shape (n, d)).
            leadopt with input_mode=fingerprint will pass numpy arrays.
            If you use input_mode=smiles, X may be a list of SMILES strings; wrapper will pass through.

        Returns
        -------
        np.ndarray
            1D array of rewards in (0,1), shape (n_samples,).
        """
        raw = self.base_model.predict(X)

        # Convert to numpy and flatten to (n,)
        raw_arr = np.asarray(raw, dtype=float).reshape(-1)

        if not np.isfinite(self.slope) or self.slope == 0.0:
            raise ValueError(f"slope must be finite and non-zero; got slope={self.slope}")

        z = (raw_arr - float(self.center)) / float(self.slope)

        # Clip z to avoid exp overflow; does not change ordering beyond saturation.
        z = np.clip(z, -float(self.clip_z), float(self.clip_z))

        return _sigmoid(z)


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(obj: Any, path: str) -> None:
    # Ensure directory exists
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _smoke_test(wrapper: LogisticSquashedRegressor) -> None:
    """
    Simple sanity check on dummy X. Works only if base_model can accept random numeric arrays.
    If your base_model expects a specific feature dimension, this may fail (that's OK).
    """
    print("[smoke] Running a simple predict() smoke test...")

    # Try a common shape (2, 2048) which matches typical ECFP bit vectors.
    X = np.zeros((2, 2048), dtype=float)
    try:
        y = wrapper.predict(X)
    except Exception as e:
        print("[smoke] Predict failed (this can be normal if your model expects different X shape).")
        print(f"[smoke] Error: {e}")
        return

    print(f"[smoke] predict(X) returned shape={y.shape}, dtype={y.dtype}")
    print(f"[smoke] values={y}")
    if np.any(~np.isfinite(y)):
        raise RuntimeError("[smoke] Non-finite outputs detected.")
    if np.any((y <= 0.0) | (y >= 1.0)):
        # Logistic is (0,1), but with extreme clipping it will be extremely close to 0/1, not equal.
        # If you see exact 0/1, it could be due to dtype/underflow issues.
        print("[smoke] Warning: outputs include <=0 or >=1; check numeric stability.")


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in_model", required=True, help="Path to base pickled regression model (predicts raw pIC50).")
    p.add_argument("--out_model", required=True, help="Path to write wrapped pickled model.")
    p.add_argument("--center", type=float, default=7.0, help="pIC50 center for sigmoid (default: 7.0).")
    p.add_argument("--slope", type=float, default=1.0, help="Sigmoid slope in pIC50 units (default: 1.0).")
    p.add_argument("--clip_z", type=float, default=30.0, help="Clip z to [-clip_z, clip_z] (default: 30).")
    p.add_argument("--smoke", action="store_true", help="Run a basic predict() smoke test after wrapping.")
    args = p.parse_args(list(argv) if argv is not None else None)

    base = _load_pickle(args.in_model)
    wrapper = LogisticSquashedRegressor(
        base_model=base,
        center=args.center,
        slope=args.slope,
        clip_z=args.clip_z,
    )
    _save_pickle(wrapper, args.out_model)

    print(f"Wrote wrapped model to: {args.out_model}")
    print(f"Wrapper: center={args.center}, slope={args.slope}, clip_z={args.clip_z}")
    print("Note: leadopt qsar_real will now maximize a bounded reward in (0,1).")

    if args.smoke:
        _smoke_test(wrapper)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())