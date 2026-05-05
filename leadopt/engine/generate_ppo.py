from __future__ import annotations

"""Shared generation engine for `leadopt generate` and `leadopt.api.generate`.

Design constraints:
- No algorithm changes: this mirrors the logic currently in leadopt.cli.generate.
- Engine returns stable dataclasses and can optionally write CSV artifacts.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from leadopt.api.types import GenerateResult, MoleculeRecord, RunMetadata
from leadopt.core.seeding import set_global_seed


def _operator_signature(operators: list[Any]) -> str:
    from leadopt.core.signatures import operator_signature

    return operator_signature(operators)


def generate_from_checkpoint(
    *,
    run_dir: Union[str, Path],
    checkpoint: Union[str, Path, None] = None,
    checkpoint_name: str = "model_best",  # "model_best" or "model_last"
    preset: str = "",
    preset_path_for_metadata: Optional[Union[str, Path]] = None,
    smiles: str,
    seed: int = 0,
    episodes: int = 128,
    top_k: int = 50,
    policy: str = "sample",  # "sample" or "greedy"
    out_csv: Optional[Union[str, Path]] = None,
    write_files: bool = True,
    device: Optional[str] = None,
) -> GenerateResult:
    run_dir_p = Path(run_dir)
    if not run_dir_p.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir_p}")

    # Resolve checkpoint path (same semantics as CLI)
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = run_dir_p / str(checkpoint)
    else:
        ckpt_file = (
            "model_best.pt" if checkpoint_name == "model_best" else "model_last.pt"
        )
        ckpt_path = run_dir_p / ckpt_file

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    import torch
    from rdkit import Chem

    from leadopt.models.action_vocab import ActionVocab
    from leadopt.models.featurizers import mol_to_graph_tensors
    from leadopt.models.mpnn_policy import MPNNPolicy
    from leadopt.rl.ppo import PPOConfig, PPOTrainer

    set_global_seed(int(seed), deterministic_torch=True)
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab
    vocab_path = run_dir_p / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab.json in run_dir: {vocab_path}")
    vocab = ActionVocab.from_json(vocab_path.read_text(encoding="utf-8"))

    # Build env: preset mode or legacy mode (same as CLI)
    use_preset = bool(str(preset).strip())
    if use_preset:
        from leadopt.cli._preset_path import resolve_preset_path as _cli_resolve
        from leadopt.config.preset_loader import PresetLoader

        with _cli_resolve(str(preset)) as preset_path:
            loaded = PresetLoader().load(Path(preset_path))
        env = PresetLoader().build_env(loaded)

        # operator compatibility check
        sig_file = run_dir_p / "operators_sig.txt"
        if sig_file.exists():
            sig_prev = sig_file.read_text(encoding="utf-8").strip()
            sig_now = _operator_signature(list(loaded.operators))
            if sig_prev != sig_now:
                raise RuntimeError(
                    "operators_sig mismatch: the preset operators do not match the training run.\n"
                    f"run_dir={run_dir_p}\n"
                    f"preset={preset}\n"
                    f"prev={sig_prev}\nnow={sig_now}"
                )
    else:
        from leadopt.config.presets import lead_optimization_preset
        from leadopt.env import GraphEnvironment

        legacy = lead_optimization_preset()
        env = GraphEnvironment(
            operators=legacy.operators,
            score_fn=legacy.score_fn,
            max_steps=legacy.max_steps,
            seed=int(seed),
            include_terminate=True,
            require_connected=True,
            reward_mode="terminal",
            rule_config=legacy.rule_config,
        )

    mol0 = Chem.MolFromSmiles(str(smiles))
    if mol0 is None:
        raise ValueError("Invalid input SMILES for generate.")
    x, _ei, ea = mol_to_graph_tensors(mol0, torch_device)
    atom_feat_dim = int(x.shape[-1])
    bond_feat_dim = int(ea.shape[-1]) if ea.numel() > 0 else 0

    # Load checkpoint FIRST so we can reconstruct the correct model architecture.
    ckpt = torch.load(ckpt_path, map_location=torch_device)

    # Model config precedence: checkpoint > preset > defaults
    default_model_config: Dict[str, Any] = {"hidden_dim": 128, "mp_steps": 3, "emb_dim": 32}

    ckpt_model_cfg: Optional[Dict[str, Any]] = None
    if isinstance(ckpt, dict):
        mc = ckpt.get("model_config", None)
        if isinstance(mc, dict):
            ckpt_model_cfg = dict(mc)

    preset_model_cfg: Optional[Dict[str, Any]] = None
    if use_preset:
        # loaded exists in preset mode; prefer model_meta if present
        mm = getattr(loaded, "model_meta", None)
        if isinstance(mm, dict) and mm:
            preset_model_cfg = dict(mm)

    model_config: Dict[str, Any] = (
        ckpt_model_cfg
        if ckpt_model_cfg is not None
        else (preset_model_cfg if preset_model_cfg is not None else dict(default_model_config))
    )

    model = MPNNPolicy(
        atom_feat_dim=atom_feat_dim,
        bond_feat_dim=bond_feat_dim,
        vocab=vocab,
        **model_config,
    ).to(torch_device)

    # Load weights + PPO config (backwards compatible with old checkpoint formats)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        cfg_dict = ckpt.get("ppo_config", None)
        cfg = (
            PPOConfig(**cfg_dict)
            if isinstance(cfg_dict, dict)
            else PPOConfig(seed=int(seed))
        )
    else:
        model.load_state_dict(ckpt)
        cfg = PPOConfig(seed=int(seed))

    trainer = PPOTrainer(env=env, model=model, cfg=cfg, device=torch_device)

    lead_can = Chem.MolToSmiles(Chem.MolFromSmiles(str(smiles)), canonical=True)

    best_by_smiles: dict[str, float] = {}
    best_payload_by_smiles: dict[str, dict[str, Any]] = {}
    for _ in range(int(episodes)):
        env.reset(lead_can)
        while not env.done:
            mol = env.state.mol
            actions, mask_np = env.available_actions()
            if not mask_np.any():
                break
            greedy = bool(policy == "greedy")
            aidx, _logp, _v = trainer._act(mol, actions, mask_np, greedy=greedy)
            env.step(aidx)

        from leadopt.api.scoring import score_to_fields

        final_smiles = Chem.MolToSmiles(env.state.mol, canonical=True)
        res = env.state.info.get("_result", None)
        obj, comps, score_md = score_to_fields(res)

        prev = best_by_smiles.get(final_smiles)
        if prev is None or float(obj) > float(prev):
            best_by_smiles[final_smiles] = float(obj)
            # keep the best scoring breakdown for this smiles (JSON-safe)
            best_payload_by_smiles[final_smiles] = {
                "components": comps,
                "scoring": score_md,
            }

    items = sorted(best_by_smiles.items(), key=lambda t: t[1], reverse=True)
    top = items[: int(top_k)]

    candidates = []
    for i, (s, sc) in enumerate(top):
        payload = best_payload_by_smiles.get(s, {})
        comps = payload.get("components", None)
        scoring_md = payload.get("scoring", {})
        candidates.append(
            MoleculeRecord(
                smiles=str(s),
                objective=float(sc),
                components=comps if isinstance(comps, dict) else None,
                metadata={
                    "rank": i + 1,
                    "scoring": scoring_md if isinstance(scoring_md, dict) else {},
                },
            )
        )

    artifacts: Dict[str, str] = {}

    out_path: Optional[Path] = None
    if bool(write_files):
        if out_csv is not None:
            out_path = Path(out_csv)
        else:
            out_path = run_dir_p / f"generated_topk_{policy}.csv"

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "smiles,score\n" + "\n".join([f"{s},{sc:.6f}" for s, sc in top]) + "\n",
            encoding="utf-8",
        )
        artifacts["csv"] = str(out_path)

    md = RunMetadata(
        seed=int(seed),
        device=device,
        preset_name=str(preset) if str(preset).strip() else None,
        preset_path=(
            str(preset_path_for_metadata)
            if preset_path_for_metadata is not None
            else None
        ),
        run_dir=str(run_dir_p),
        extra={
            "generate": {
                "episodes": int(episodes),
                "top_k": int(top_k),
                "policy": str(policy),
                "checkpoint": str(ckpt_path),
                "model_config": dict(model_config),
            }
        },
    )

    return GenerateResult(
        lead=MoleculeRecord(smiles=lead_can, objective=0.0, metadata={"role": "lead"}),
        unique_count=len(best_by_smiles),
        candidates=candidates,
        metadata=md,
        artifacts=artifacts,
    )
