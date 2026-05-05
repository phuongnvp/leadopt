from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _require_optional_deps(dataset_mode: bool) -> None:
    """Raise a helpful error if heavy optional deps are missing."""
    missing: List[str] = []
    try:
        import torch  # noqa: F401
    except Exception:
        missing.append("torch (pip install 'leadopt[rl]')")
    try:
        import rdkit  # noqa: F401
    except Exception:
        missing.append("rdkit (pip install 'leadopt[chem]')")
    if dataset_mode:
        try:
            import pandas  # noqa: F401
        except Exception:
            missing.append("pandas")
        try:
            import sklearn  # noqa: F401
        except Exception:
            missing.append("scikit-learn")
    try:
        import tqdm  # noqa: F401
    except Exception:
        missing.append("tqdm")
    if missing:
        raise RuntimeError(
            "leadopt train requires optional dependencies that are not installed: "
            + ", ".join(missing)
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="leadopt train",
        description=(
            "Train a PPO policy for lead optimization. Supports either a dataset of leads "
            "(--dataset/--smiles_col) or a single lead (--smiles)."
        ),
    )

    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for run artifacts/checkpoints.",
    )
    ap.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="(alias) Output directory (legacy name).",
    )
    ap.add_argument(
        "--preset",
        type=str,
        default="",
        help=(
            "Optional YAML preset path or shipped preset name. If empty, uses the legacy QED preset."
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional: override seed (otherwise from preset or 0).",
    )

    # Input: dataset or single lead
    in_group = ap.add_mutually_exclusive_group(required=True)
    in_group.add_argument(
        "--smiles",
        type=str,
        default=None,
        help="Single lead SMILES to optimize (will be canonicalized).",
    )
    in_group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="CSV dataset of lead SMILES.",
    )
    ap.add_argument(
        "--smiles_col",
        type=str,
        default="smiles",
        help="CSV column name for SMILES (default: smiles).",
    )
    ap.add_argument(
        "--dataset_limit",
        type=int,
        default=0,
        help="Optional: limit number of leads loaded from CSV (default: 0 = no limit).",
    )
    ap.add_argument(
        "--split_test_size",
        type=float,
        default=0.2,
        help="Train/eval split fraction (default: 0.2).",
    )
    ap.add_argument(
        "--split_random_state",
        type=int,
        default=0,
        help="Train/eval split RNG seed (default: 0).",
    )

    # Training loop controls
    ap.add_argument("--total_updates", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--eval_episodes_per_lead", type=int, default=8)
    ap.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save a periodic checkpoint every N updates.",
    )
    ap.add_argument(
        "--keep_last_k",
        type=int,
        default=5,
        help="Keep last K periodic checkpoints. Use 0 to keep all.",
    )
    ap.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Path to checkpoint to resume from. If empty, tries out_dir/model_last.pt if present."
        ),
    )
    return ap


# ----------------------------
# Shared utilities (copied from scripts/train.py and scripts/single_train.py)
# ----------------------------


def _canonicalize(smiles: str) -> Optional[str]:
    from leadopt.core.smiles import canonicalize_smiles

    return canonicalize_smiles(smiles)


def _operator_signature(operators: List[Any]) -> str:
    from leadopt.core.signatures import operator_signature

    return operator_signature(operators)


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _infer_feat_dims(smiles: str, device: Any) -> Tuple[int, int]:
    from rdkit import Chem

    from leadopt.models.featurizers import mol_to_graph_tensors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for feature dim inference: {smiles}")
    x, _edge_index, edge_attr = mol_to_graph_tensors(mol, device)
    atom_dim = int(x.shape[-1])
    bond_dim = int(edge_attr.shape[-1]) if edge_attr.numel() > 0 else 0
    return atom_dim, bond_dim


def _objective_from_env_scorer(env: Any, smiles: str) -> float:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    try:
        res = env.scorer.score(mol, context=None)
        return float(res.objective)
    except Exception:
        return 0.0


def _run_one_greedy_episode(
    env: Any, trainer: Any, lead_smiles: str
) -> Tuple[str, float, List[Any]]:
    from rdkit import Chem

    env.reset(lead_smiles)
    while not env.done:
        mol = env.state.mol
        actions, mask_np = env.available_actions()
        assert mask_np.any(), "No allowed actions; action space broken."
        aidx, _logp, _v = trainer._act(mol, actions, mask_np, greedy=True)
        env.step(aidx)

    final_smiles = Chem.MolToSmiles(env.state.mol, canonical=True)
    res = env.state.info.get("_result", None)
    final_score = float(getattr(res, "objective", 0.0))
    traj_actions = env.state.info.get("trajectory", [])
    return final_smiles, final_score, traj_actions


def _make_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    trainer: Any,
    update_idx: int,
    cfg: Any,
    best_score: float,
    np_rng: Any,
    model_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import torch

    ckpt: Dict[str, Any] = {
        "update": int(update_idx),
        "best_score": float(best_score),
        "ppo_config": asdict(cfg),
        "model_config": dict(model_config) if model_config is not None else None,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np_rng.bit_generator.state,
        "trainer_rng_state": trainer.rng.bit_generator.state,
    }
    if torch.cuda.is_available():
        ckpt["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return ckpt


def _save_checkpoint(path: Path, ckpt: Dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def _load_checkpoint(path: Path, device: Any) -> Dict[str, Any]:
    import torch

    return torch.load(path, map_location=device)


def _restore_rng_states(ckpt: Dict[str, Any], *, trainer: Any, np_rng: Any) -> None:
    import torch

    if "torch_rng_state" in ckpt:
        torch.set_rng_state(ckpt["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in ckpt:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
    if "numpy_rng_state" in ckpt:
        np_rng.bit_generator.state = ckpt["numpy_rng_state"]
    if "trainer_rng_state" in ckpt:
        trainer.rng.bit_generator.state = ckpt["trainer_rng_state"]


# ----------------------------
# Dataset mode (ported from scripts/train.py)
# ----------------------------


def _load_smiles_from_csv(
    csv_path: str, smiles_col: str, *, limit: int = 0
) -> List[str]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise KeyError(
            f"Column '{smiles_col}' not found in {csv_path}. Columns: {list(df.columns)}"
        )

    raw = df[smiles_col].dropna().astype(str).tolist()
    if limit and limit > 0:
        raw = raw[:limit]

    cleaned: List[str] = []
    for s in raw:
        c = _canonicalize(s)
        if c:
            cleaned.append(c)

    # Deterministic de-dup while preserving order
    seen = set()
    unique: List[str] = []
    for s in cleaned:
        if s not in seen:
            unique.append(s)
            seen.add(s)

    if len(unique) == 0:
        raise ValueError("No valid SMILES found after cleaning/canonicalization.")
    return unique


def _make_split(
    unique_smiles: List[str], *, test_size: float, random_state: int
) -> Tuple[List[str], List[str]]:
    from sklearn.model_selection import train_test_split

    train_leads, eval_leads = train_test_split(
        unique_smiles,
        test_size=float(test_size),
        random_state=int(random_state),
        shuffle=True,
    )
    return list(train_leads), list(eval_leads)


def _score_fn_qed(smiles: str) -> float:
    from rdkit import Chem
    from rdkit.Chem import QED

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    try:
        return float(QED.qed(mol))
    except Exception:
        return 0.0


def _train_dataset_mode(args: argparse.Namespace, *, out_dir: Path) -> None:
    import numpy as np
    import torch
    from tqdm import trange

    from leadopt.cli._preset_path import resolve_preset_path
    from leadopt.config.preset_loader import PresetLoader, PresetLoaderError
    from leadopt.config.presets import lead_optimization_preset
    from leadopt.core.run_logging import write_run_artifacts
    from leadopt.core.seeding import set_global_seed
    from leadopt.env import GraphEnvironment
    from leadopt.models.action_vocab import ActionVocab
    from leadopt.models.mpnn_policy import MPNNPolicy
    from leadopt.rl.ppo import PPOConfig, PPOTrainer
    from leadopt.sar import SARLogger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_yaml = bool(str(args.preset).strip())
    loader: Optional[PresetLoader] = None
    loaded = None

    if use_yaml:
        loader = PresetLoader()
        try:
            with resolve_preset_path(str(args.preset)) as pp:
                loaded = loader.load(Path(pp))
        except PresetLoaderError as e:
            raise RuntimeError(f"Failed to load preset YAML: {args.preset}\n{e}") from e

    # Seed: CLI override > YAML > legacy
    if args.seed is not None:
        seed_used = int(args.seed)
    elif use_yaml and loaded is not None:
        seed_used = int(loaded.training_meta.get("seed", 0))
    else:
        seed_used = 0

    set_global_seed(seed_used, deterministic_torch=True)
    np_rng = np.random.default_rng(seed_used)

    # YAML reproducibility artifacts
    if use_yaml and loaded is not None:
        preset_used_path = out_dir / "preset_used.yaml"
        manifest_path = out_dir / "run_manifest.json"
        if (not preset_used_path.exists()) or (not manifest_path.exists()):
            write_run_artifacts(
                run_dir=out_dir,
                preset_source_path=loaded.preset.source_path,
                preset_raw=loaded.preset.raw,
                preset_name=loaded.preset.raw.get("name", "unknown"),
                seed=seed_used,
                scorer=loaded.scorer,
                constraint_suite=loaded.constraint_suite,
                reward_composer=loaded.reward_composer,
                env_kwargs=loaded.env_kwargs,
                training_meta=loaded.training_meta,
                extra={
                    "cli": {
                        "out_dir": str(out_dir),
                        "preset": str(args.preset),
                        "seed_override": args.seed,
                    },
                    "model": {
                        "type": "MPNNPolicy",
                        "params": dict(getattr(loaded, "model_meta", {})),
                    },
                    "dataset": {
                        "path": str(args.dataset),
                        "smiles_col": str(args.smiles_col),
                        "limit": int(args.dataset_limit),
                        "split_test_size": float(args.split_test_size),
                        "split_random_state": int(args.split_random_state),
                    },
                },
            )

    # Resume detection
    resume_path: Optional[Path] = None
    if str(args.resume).strip():
        resume_path = Path(str(args.resume))
    else:
        candidate = out_dir / "model_last.pt"
        if candidate.exists():
            resume_path = candidate

    # Run config (dataset + split) must be stable across resume
    run_config_path = out_dir / "run_config.json"
    run_cfg: Dict[str, Any] = (
        _load_json(run_config_path) if run_config_path.exists() else {}
    )
    is_resuming = resume_path is not None

    if is_resuming:
        if "train_leads" not in run_cfg or "eval_leads" not in run_cfg:
            raise RuntimeError(
                "Resuming requires run_config.json to contain 'train_leads' and 'eval_leads'. "
                "(Did you delete run_config.json?)"
            )
        train_leads = list(run_cfg["train_leads"])
        eval_leads = list(run_cfg["eval_leads"])
        if not train_leads or not eval_leads:
            raise RuntimeError(
                "Resuming found empty train/eval leads in run_config.json"
            )
        print(f"Resume mode: loaded train/eval leads from {run_config_path}")
    else:
        unique = _load_smiles_from_csv(
            str(args.dataset),
            str(args.smiles_col),
            limit=int(args.dataset_limit),
        )
        train_leads, eval_leads = _make_split(
            unique,
            test_size=float(args.split_test_size),
            random_state=int(args.split_random_state),
        )
        if not train_leads or not eval_leads:
            raise RuntimeError("Split produced empty train/eval leads")
        print("Fresh mode: loaded dataset leads")
        print(f"Train leads: {len(train_leads)} | Eval leads: {len(eval_leads)}")

    # Env + vocab
    if use_yaml and loaded is not None and loader is not None:
        env = loader.build_env(loaded)
        operators_for_vocab = loaded.operators
    else:
        preset = lead_optimization_preset()
        env = GraphEnvironment(
            operators=preset.operators,
            score_fn=_score_fn_qed,
            max_steps=preset.max_steps,
            seed=seed_used,
            include_terminate=True,
            require_connected=True,
            reward_mode="terminal",
            rule_config=preset.rule_config,
        )
        operators_for_vocab = preset.operators

    vocab_path = out_dir / "vocab.json"
    sig_path = out_dir / "operators_sig.txt"
    sig_now = _operator_signature(operators_for_vocab)
    if sig_path.exists():
        sig_prev = sig_path.read_text(encoding="utf-8").strip()
        if sig_prev != sig_now:
            raise RuntimeError(
                "Operator set changed since this run was created (operators_sig mismatch). "
                "Refusing to proceed because vocab/action-space may be incompatible.\n"
                f"prev={sig_prev}\nnow={sig_now}"
            )
    else:
        sig_path.write_text(sig_now, encoding="utf-8")


    if vocab_path.exists():
        vocab = ActionVocab.from_json(vocab_path.read_text())
    else:
        vocab = ActionVocab.build(operators_for_vocab, include_terminate=True)
        vocab_path.write_text(vocab.to_json())

    atom_feat_dim, bond_feat_dim = _infer_feat_dims(train_leads[0], device)

    # Model hyperparameters: defaults unless YAML provides overrides.
    if use_yaml and loaded is not None:
        model_config: Dict[str, Any] = dict(getattr(loaded, "model_meta", {}))
        if not model_config:
            model_config = {"hidden_dim": 128, "mp_steps": 3, "emb_dim": 32}
    else:
        # Preserve legacy behavior for non-YAML runs.
        model_config = {"hidden_dim": 128, "mp_steps": 3, "emb_dim": 32}

    # If resuming, prefer checkpoint-stored model_config for reproducibility.
    resume_ckpt: Optional[Dict[str, Any]] = None
    if resume_path is not None and resume_path.exists():
        resume_ckpt = _load_checkpoint(resume_path, device=device)
        ckpt_model_cfg = resume_ckpt.get("model_config", None)
        if isinstance(ckpt_model_cfg, dict):
            if use_yaml and loaded is not None:
                if dict(ckpt_model_cfg) != dict(model_config):
                    print(
                        "WARNING: Resuming checkpoint contains a different model_config than the preset. "
                        "Using checkpoint model_config for reproducibility."
                    )
            model_config = dict(ckpt_model_cfg)

    model = MPNNPolicy(
        atom_feat_dim=atom_feat_dim,
        bond_feat_dim=bond_feat_dim,
        vocab=vocab,
        **model_config,
    ).to(device)

    if use_yaml and loaded is not None:
        cfg = loaded.ppo_config
        cfg.seed = int(seed_used)
    else:
        cfg = PPOConfig(
            gamma=0.99,
            lam=0.95,
            rollout_episodes=16,
            update_epochs=2,
            minibatch_size=64,
            lr=3e-4,
            ent_coef=0.01,
            seed=seed_used,
        )

    trainer = PPOTrainer(env=env, model=model, cfg=cfg, device=device)
    logger = SARLogger()

    # Resume state
    start_update = 0
    best_score = -1.0
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume path does not exist: {resume_path}")
        ckpt = resume_ckpt if resume_ckpt is not None else _load_checkpoint(
            resume_path, device=device
        )
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        if "optimizer_state_dict" in ckpt:
            trainer.opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_update = int(ckpt.get("update", -1)) + 1
        best_score = float(ckpt.get("best_score", -1.0))
        _restore_rng_states(ckpt, trainer=trainer, np_rng=np_rng)
        print(f"\nResumed from: {resume_path}")
        print(f"Start update: {start_update}")
        print(f"Best eval so far: {best_score:.4f}\n")

    # Save run metadata on fresh runs
    if not run_config_path.exists():
        run_cfg = {
            "seed": seed_used,
            "device": str(device),
            "train_leads": train_leads,
            "eval_leads": eval_leads,
            "ppo_config": asdict(cfg),
            "model_config": dict(model_config),
            "preset": (
                str(Path(str(args.preset))) if use_yaml else "lead_optimization_preset"
            ),
            "objective": ("preset_scorer" if use_yaml else "QED"),
            "atom_feat_dim": atom_feat_dim,
            "bond_feat_dim": bond_feat_dim,
        }
        _save_json(run_config_path, run_cfg)

    periodic_ckpts: List[Path] = sorted(out_dir.glob("checkpoint_update_*.pt"))

    for upd in trange(start_update, int(args.total_updates), desc="PPO updates"):
        lead_smiles = str(np_rng.choice(train_leads))
        traj = trainer.collect_rollout(lead_smiles)
        stats = trainer.update(traj)

        if int(args.save_every) > 0 and (upd + 1) % int(args.save_every) == 0:
            ckpt = _make_checkpoint(
                model=model,
                optimizer=trainer.opt,
                trainer=trainer,
                update_idx=upd,
                cfg=cfg,
                best_score=best_score,
                np_rng=np_rng,
                model_config=model_config,
            )
            path = out_dir / f"checkpoint_update_{upd:05d}.pt"
            _save_checkpoint(path, ckpt)
            periodic_ckpts.append(path)
            if int(args.keep_last_k) and int(args.keep_last_k) > 0:
                while len(periodic_ckpts) > int(args.keep_last_k):
                    old = periodic_ckpts.pop(0)
                    if old.exists():
                        old.unlink()

        if (upd + 1) % int(args.eval_every) == 0 or upd == start_update:
            import numpy as np

            eval_scores: List[float] = []
            for lead in eval_leads:
                scores = []
                for _ in range(int(args.eval_episodes_per_lead)):
                    _fs, sc, _tr = _run_one_greedy_episode(env, trainer, lead)
                    scores.append(sc)
                eval_scores.append(float(np.mean(scores)))
            mean_eval = float(np.mean(eval_scores)) if eval_scores else 0.0

            eid = logger.start_episode()
            if use_yaml:
                lead_score = _objective_from_env_scorer(env, lead_smiles)
            else:
                lead_score = _score_fn_qed(lead_smiles)
            final_smiles, final_score, traj_actions = _run_one_greedy_episode(
                env, trainer, lead_smiles
            )
            logger.log_episode(
                episode_id=eid,
                lead_smiles=lead_smiles,
                final_smiles=final_smiles,
                lead_score=float(lead_score),
                final_score=float(final_score),
                trajectory=traj_actions,
            )

            if mean_eval > best_score:
                best_score = float(mean_eval)
                ckpt_best = _make_checkpoint(
                    model=model,
                    optimizer=trainer.opt,
                    trainer=trainer,
                    update_idx=upd,
                    cfg=cfg,
                    best_score=best_score,
                    np_rng=np_rng,
                    model_config=model_config,
                )
                _save_checkpoint(out_dir / "model_best.pt", ckpt_best)

            ckpt_last = _make_checkpoint(
                model=model,
                optimizer=trainer.opt,
                trainer=trainer,
                update_idx=upd,
                cfg=cfg,
                best_score=best_score,
                np_rng=np_rng,
                model_config=model_config,
            )
            _save_checkpoint(out_dir / "model_last.pt", ckpt_last)

            print(
                f"\nUpdate {upd:05d} | "
                f"TrainLead {lead_smiles} | "
                f"GreedyScore {final_score:.4f} | "
                f"EvalMean {mean_eval:.4f} | "
                f"PiLoss {stats['loss_pi']:.4f} | "
                f"VLoss {stats['loss_v']:.4f} | "
                f"Ent {stats['entropy']:.4f} | "
                f"BestEval {best_score:.4f}"
            )

    # ----------------------------
    # SAR report (matches scripts/train.py behavior)
    # ----------------------------
    from leadopt.sar import SARAnalyzer

    analyzer = SARAnalyzer(logger.records)
    paths = analyzer.write_report(out_dir / "sar_report", top_k_sdf=50)

    print("\nWrote SAR outputs:")
    for k, p in paths.items():
        print(f"  {k}: {p}")

    print(f"\nDone. Outputs in: {out_dir}")


# ----------------------------
# Single-lead mode (ported from scripts/single_train.py)
# ----------------------------


def _train_single_mode(args: argparse.Namespace, *, out_dir: Path) -> None:
    import numpy as np
    import torch
    from tqdm import trange

    from leadopt.cli._preset_path import resolve_preset_path
    from leadopt.config.preset_loader import PresetLoader, PresetLoaderError
    from leadopt.config.presets import lead_optimization_preset
    from leadopt.core.run_logging import write_run_artifacts
    from leadopt.core.seeding import set_global_seed
    from leadopt.env import GraphEnvironment
    from leadopt.models.action_vocab import ActionVocab
    from leadopt.models.mpnn_policy import MPNNPolicy
    from leadopt.rl.ppo import PPOConfig, PPOTrainer
    from leadopt.sar import SARLogger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_yaml = bool(str(args.preset).strip())
    loader: Optional[PresetLoader] = None
    loaded = None

    if use_yaml:
        loader = PresetLoader()
        try:
            with resolve_preset_path(str(args.preset)) as pp:
                loaded = loader.load(Path(pp))
        except PresetLoaderError as e:
            raise RuntimeError(f"Failed to load preset YAML: {args.preset}\n{e}") from e

    if args.seed is not None:
        seed_used = int(args.seed)
    elif use_yaml and loaded is not None:
        seed_used = int(loaded.training_meta.get("seed", 0))
    else:
        seed_used = 0

    set_global_seed(seed_used, deterministic_torch=True)
    np_rng = np.random.default_rng(seed_used)

    if use_yaml and loaded is not None:
        preset_used_path = out_dir / "preset_used.yaml"
        manifest_path = out_dir / "run_manifest.json"
        if (not preset_used_path.exists()) or (not manifest_path.exists()):
            write_run_artifacts(
                run_dir=out_dir,
                preset_source_path=loaded.preset.source_path,
                preset_raw=loaded.preset.raw,
                preset_name=loaded.preset.raw.get("name", "unknown"),
                seed=seed_used,
                scorer=loaded.scorer,
                constraint_suite=loaded.constraint_suite,
                reward_composer=loaded.reward_composer,
                env_kwargs=loaded.env_kwargs,
                training_meta=loaded.training_meta,
                extra={
                    "cli": {
                        "out_dir": str(out_dir),
                        "preset": str(args.preset),
                        "seed_override": args.seed,
                    },
                    "model": {
                        "type": "MPNNPolicy",
                        "params": dict(getattr(loaded, "model_meta", {})),
                    },
                    "single_input": {"smiles": str(args.smiles)},
                },
            )

    resume_path: Optional[Path] = None
    if str(args.resume).strip():
        resume_path = Path(str(args.resume))
    else:
        candidate = out_dir / "model_last.pt"
        if candidate.exists():
            resume_path = candidate

    run_config_path = out_dir / "run_config.json"
    run_cfg: Dict[str, Any] = (
        _load_json(run_config_path) if run_config_path.exists() else {}
    )
    is_resuming = resume_path is not None

    if is_resuming:
        if "smiles" not in run_cfg:
            raise RuntimeError(
                "Resuming requires run_config.json to contain 'smiles'. (Did you delete run_config.json?)"
            )
        lead_smiles_c = str(run_cfg["smiles"])
        train_leads = [lead_smiles_c]
        eval_leads = [lead_smiles_c]
        print(f"Resume mode: loaded smiles from {run_config_path}")
        print(f"SMILES: {lead_smiles_c}")
    else:
        lead_smiles_c = _canonicalize(str(args.smiles))
        if not lead_smiles_c:
            raise ValueError(f"Invalid SMILES provided: {args.smiles}")
        train_leads = [lead_smiles_c]
        eval_leads = [lead_smiles_c]
        print("Fresh mode: using single SMILES input")
        print(f"SMILES: {lead_smiles_c}")

    if use_yaml and loaded is not None and loader is not None:
        env = loader.build_env(loaded)
        operators_for_vocab = loaded.operators
    else:
        preset = lead_optimization_preset()
        env = GraphEnvironment(
            operators=preset.operators,
            score_fn=_score_fn_qed,
            max_steps=preset.max_steps,
            seed=seed_used,
            include_terminate=True,
            require_connected=True,
            reward_mode="terminal",
            rule_config=preset.rule_config,
        )
        operators_for_vocab = preset.operators

    vocab_path = out_dir / "vocab.json"
    sig_path = out_dir / "operators_sig.txt"
    sig_now = _operator_signature(operators_for_vocab)
    if sig_path.exists():
        sig_prev = sig_path.read_text(encoding="utf-8").strip()
        if sig_prev != sig_now:
            raise RuntimeError(
                "Operator set changed since this run was created (operators_sig mismatch). "
                "Refusing to proceed because vocab/action-space may be incompatible.\n"
                f"prev={sig_prev}\nnow={sig_now}"
            )
    else:
        sig_path.write_text(sig_now, encoding="utf-8")

    if vocab_path.exists():
        vocab = ActionVocab.from_json(vocab_path.read_text())
    else:
        vocab = ActionVocab.build(operators_for_vocab, include_terminate=True)
        vocab_path.write_text(vocab.to_json())

    atom_feat_dim, bond_feat_dim = _infer_feat_dims(train_leads[0], device)

    # Model hyperparameters: defaults unless YAML provides overrides.
    if use_yaml and loaded is not None:
        model_config: Dict[str, Any] = dict(getattr(loaded, "model_meta", {}))
        if not model_config:
            model_config = {"hidden_dim": 128, "mp_steps": 3, "emb_dim": 32}
    else:
        # Preserve legacy behavior for non-YAML runs.
        model_config = {"hidden_dim": 128, "mp_steps": 3, "emb_dim": 32}

    # If resuming, prefer checkpoint-stored model_config for reproducibility.
    resume_ckpt: Optional[Dict[str, Any]] = None
    if resume_path is not None and resume_path.exists():
        resume_ckpt = _load_checkpoint(resume_path, device=device)
        ckpt_model_cfg = resume_ckpt.get("model_config", None)
        if isinstance(ckpt_model_cfg, dict):
            if use_yaml and loaded is not None:
                if dict(ckpt_model_cfg) != dict(model_config):
                    print(
                        "WARNING: Resuming checkpoint contains a different model_config than the preset. "
                        "Using checkpoint model_config for reproducibility."
                    )
            model_config = dict(ckpt_model_cfg)

    model = MPNNPolicy(
        atom_feat_dim=atom_feat_dim,
        bond_feat_dim=bond_feat_dim,
        vocab=vocab,
        **model_config,
    ).to(device)

    if use_yaml and loaded is not None:
        cfg = loaded.ppo_config
        cfg.seed = int(seed_used)
    else:
        from leadopt.rl.ppo import PPOConfig

        cfg = PPOConfig(
            gamma=0.99,
            lam=0.95,
            rollout_episodes=16,
            update_epochs=2,
            minibatch_size=64,
            lr=3e-4,
            ent_coef=0.01,
            seed=seed_used,
        )


    trainer = PPOTrainer(env=env, model=model, cfg=cfg, device=device)
    logger = SARLogger()

    start_update = 0
    best_score = -1.0
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume path does not exist: {resume_path}")
        ckpt = resume_ckpt if resume_ckpt is not None else _load_checkpoint(
            resume_path, device=device
        )
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        if "optimizer_state_dict" in ckpt:
            trainer.opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_update = int(ckpt.get("update", -1)) + 1
        best_score = float(ckpt.get("best_score", -1.0))
        _restore_rng_states(ckpt, trainer=trainer, np_rng=np_rng)
        print(f"\nResumed from: {resume_path}")
        print(f"Start update: {start_update}")
        print(f"Best eval so far: {best_score:.4f}\n")

    if not run_config_path.exists():
        run_cfg = {
            "seed": seed_used,
            "device": str(device),
            "train_leads": train_leads,
            "eval_leads": eval_leads,
            "ppo_config": asdict(cfg),
            "model_config": dict(model_config),
            "preset": (
                str(Path(str(args.preset))) if use_yaml else "lead_optimization_preset"
            ),
            "objective": ("preset_scorer" if use_yaml else "QED"),
            "atom_feat_dim": atom_feat_dim,
            "bond_feat_dim": bond_feat_dim,
        }
        _save_json(run_config_path, run_cfg)

    periodic_ckpts: List[Path] = sorted(out_dir.glob("checkpoint_update_*.pt"))
    for upd in trange(start_update, int(args.total_updates), desc="PPO updates"):
        lead_smiles = str(np_rng.choice(train_leads))
        traj = trainer.collect_rollout(lead_smiles)
        stats = trainer.update(traj)

        if int(args.save_every) > 0 and (upd + 1) % int(args.save_every) == 0:
            ckpt = _make_checkpoint(
                model=model,
                optimizer=trainer.opt,
                trainer=trainer,
                update_idx=upd,
                cfg=cfg,
                best_score=best_score,
                np_rng=np_rng,
                model_config=model_config,
            )
            path = out_dir / f"checkpoint_update_{upd:05d}.pt"
            _save_checkpoint(path, ckpt)
            periodic_ckpts.append(path)
            if int(args.keep_last_k) and int(args.keep_last_k) > 0:
                while len(periodic_ckpts) > int(args.keep_last_k):
                    old = periodic_ckpts.pop(0)
                    if old.exists():
                        old.unlink()

        if (upd + 1) % int(args.eval_every) == 0 or upd == start_update:
            import numpy as np

            scores = []
            for _ in range(int(args.eval_episodes_per_lead)):
                _fs, sc, _tr = _run_one_greedy_episode(env, trainer, train_leads[0])
                scores.append(sc)
            mean_eval = float(np.mean(scores)) if scores else 0.0

            eid = logger.start_episode()
            if use_yaml:
                lead_score = _objective_from_env_scorer(env, lead_smiles)
            else:
                lead_score = _score_fn_qed(lead_smiles)
            final_smiles, final_score, traj_actions = _run_one_greedy_episode(
                env, trainer, lead_smiles
            )
            logger.log_episode(
                episode_id=eid,
                lead_smiles=lead_smiles,
                final_smiles=final_smiles,
                lead_score=float(lead_score),
                final_score=float(final_score),
                trajectory=traj_actions,
            )

            if mean_eval > best_score:
                best_score = float(mean_eval)
                ckpt_best = _make_checkpoint(
                    model=model,
                    optimizer=trainer.opt,
                    trainer=trainer,
                    update_idx=upd,
                    cfg=cfg,
                    best_score=best_score,
                    np_rng=np_rng,
                    model_config=model_config,
                )
                _save_checkpoint(out_dir / "model_best.pt", ckpt_best)

            ckpt_last = _make_checkpoint(
                model=model,
                optimizer=trainer.opt,
                trainer=trainer,
                update_idx=upd,
                cfg=cfg,
                best_score=best_score,
                np_rng=np_rng,
                model_config=model_config,
            )
            _save_checkpoint(out_dir / "model_last.pt", ckpt_last)

            print(
                f"\nUpdate {upd:05d} | "
                f"Lead {lead_smiles} | "
                f"GreedyScore {final_score:.4f} | "
                f"EvalMean {mean_eval:.4f} | "
                f"PiLoss {stats['loss_pi']:.4f} | "
                f"VLoss {stats['loss_v']:.4f} | "
                f"Ent {stats['entropy']:.4f} | "
                f"BestEval {best_score:.4f}"
            )

    # ----------------------------
    # SAR report (matches scripts/train.py behavior)
    # ----------------------------
    from leadopt.sar import SARAnalyzer

    analyzer = SARAnalyzer(logger.records)
    paths = analyzer.write_report(out_dir / "sar_report", top_k_sdf=50)

    print("\nWrote SAR outputs:")
    for k, p in paths.items():
        print(f"  {k}: {p}")

    print(f"\nDone. Outputs in: {out_dir}")


def main(argv: Optional[list[str]] = None) -> None:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)

    out_dir = args.out_dir or args.run_dir
    if not out_dir:
        raise SystemExit("Missing required argument: --out_dir (or legacy --run_dir)")
    out_dir_p = Path(str(out_dir))
    out_dir_p.mkdir(parents=True, exist_ok=True)

    dataset_mode = args.dataset is not None
    _require_optional_deps(dataset_mode=dataset_mode)

    from leadopt.engine.train_ppo import train_dataset, train_single

    if dataset_mode:
        train_dataset(
            run_dir=out_dir_p,
            preset=str(args.preset),
            seed=args.seed,
            dataset=Path(str(args.dataset)),
            smiles_col=str(args.smiles_col),
            dataset_limit=int(args.dataset_limit),
            split_test_size=float(args.split_test_size),
            split_random_state=int(args.split_random_state),
            total_updates=int(args.total_updates),
            eval_every=int(args.eval_every),
            eval_episodes_per_lead=int(args.eval_episodes_per_lead),
            save_every=int(args.save_every),
            keep_last_k=int(args.keep_last_k),
            resume_from=str(args.resume) if str(args.resume).strip() else None,
        )
    else:
        train_single(
            run_dir=out_dir_p,
            preset=str(args.preset),
            seed=args.seed,
            smiles=str(args.smiles),
            total_updates=int(args.total_updates),
            eval_every=int(args.eval_every),
            eval_episodes_per_lead=int(args.eval_episodes_per_lead),
            save_every=int(args.save_every),
            keep_last_k=int(args.keep_last_k),
            resume_from=str(args.resume) if str(args.resume).strip() else None,
        )
