from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

# -----------------------------------------------------------------------------
# Simple timing registry
# -----------------------------------------------------------------------------
_TIMES: Dict[str, float] = defaultdict(float)
_CALLS: Dict[str, int] = defaultdict(int)


def _wrap_method(obj: Any, name: str, label: str) -> None:
    """Monkeypatch obj.name to time it."""
    if not hasattr(obj, name):
        return
    orig = getattr(obj, name)
    if not callable(orig):
        return

    def wrapped(*args: Any, **kwargs: Any):
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            _TIMES[label] += time.perf_counter() - t0
            _CALLS[label] += 1

    setattr(obj, name, wrapped)


def _wrap_callable(fn: Callable[..., Any], label: str) -> Callable[..., Any]:
    """Wrap a function to time it."""
    def wrapped(*args: Any, **kwargs: Any):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _TIMES[label] += time.perf_counter() - t0
            _CALLS[label] += 1
    return wrapped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=str, default="medchem_quality_tier4")
    ap.add_argument("--smiles", type=str, default="CCO")
    ap.add_argument("--run_dir", type=str, default="runs/instrument_api_single")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total_updates", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--eval_episodes_per_lead", type=int, default=2)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--keep_last_k", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------------
    # Monkeypatch key hot methods *at class level* so they affect training path.
    # This avoids editing leadopt code.
    # -----------------------------------------------------------------------------
    import leadopt.rl.ppo as ppo_mod
    import leadopt.env as env_mod

    # PPOTrainer hotspots
    if hasattr(ppo_mod, "PPOTrainer"):
        _wrap_method(ppo_mod.PPOTrainer, "collect_rollout", "ppo.collect_rollout")
        _wrap_method(ppo_mod.PPOTrainer, "update", "ppo.update")
        _wrap_method(ppo_mod.PPOTrainer, "_act", "ppo._act")

    # Environment hotspots (GraphEnvironment typically)
    if hasattr(env_mod, "GraphEnvironment"):
        _wrap_method(env_mod.GraphEnvironment, "available_actions", "env.available_actions")
        _wrap_method(env_mod.GraphEnvironment, "step", "env.step")
        _wrap_method(env_mod.GraphEnvironment, "reset", "env.reset")

    # Featurization hotspot (if present as function)
    try:
        import leadopt.models.featurizers as feat_mod
        if hasattr(feat_mod, "mol_to_graph_tensors"):
            feat_mod.mol_to_graph_tensors = _wrap_callable(
                feat_mod.mol_to_graph_tensors, "feat.mol_to_graph_tensors"
            )
    except Exception:
        pass

    # -----------------------------------------------------------------------------
    # Run a tiny training via API (uses engine -> cli internals).
    # -----------------------------------------------------------------------------
    from leadopt.api import train

    t0 = time.perf_counter()
    tr = train(
        preset=str(args.preset),
        smiles=str(args.smiles),
        run_dir=run_dir,
        seed=int(args.seed),
        total_updates=int(args.total_updates),
        eval_every=int(args.eval_every),
        eval_episodes_per_lead=int(args.eval_episodes_per_lead),
        save_every=int(args.save_every),
        keep_last_k=int(args.keep_last_k),
        write_artifacts=False,
    )
    total = time.perf_counter() - t0

    # -----------------------------------------------------------------------------
    # Print timing report
    # -----------------------------------------------------------------------------
    print("\n=== TrainResult ===")
    print("run_dir:", tr.run_dir)
    print("best_checkpoint:", tr.best_checkpoint)
    print("last_checkpoint:", tr.last_checkpoint)

    print("\n=== Timing summary (wall time) ===")
    print(f"TOTAL: {total:.3f} s\n")

    rows: list[Tuple[str, float, int, float]] = []
    for k, sec in _TIMES.items():
        calls = _CALLS.get(k, 0)
        avg_ms = (sec / calls * 1000.0) if calls else 0.0
        rows.append((k, sec, calls, avg_ms))
    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"{'section':30s} {'time(s)':>12s} {'calls':>10s} {'avg(ms)':>12s}")
    print("-" * 70)
    for k, sec, calls, avg_ms in rows:
        print(f"{k:30s} {sec:12.3f} {calls:10d} {avg_ms:12.3f}")

    print("\nTip: The largest section by time(s) is your primary bottleneck.")


if __name__ == "__main__":
    main()