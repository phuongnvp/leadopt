from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

from ..actions import ActionInstance
from ..env import GraphEnvironment
from ..models.mpnn_policy import (
    MPNNPolicy,
    masked_categorical_sample,
    masked_entropy,
    masked_logprob,
)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 1.0

    rollout_episodes: int = 64
    update_epochs: int = 4
    minibatch_size: int = 64

    # Reproducibility: seed PPO's own RNG for minibatch shuffling, etc.
    seed: int = 0


@dataclass
class Transition:
    mol: Chem.Mol
    actions: List[ActionInstance]
    mask: torch.Tensor  # [A] bool (CPU ok)
    act_idx: int
    logp_old: float
    value_old: float
    reward: float
    done: bool


class PPOTrainer:
    def __init__(
        self,
        env: GraphEnvironment,
        model: MPNNPolicy,
        cfg: PPOConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.env = env
        self.model = model
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        # PPO-local RNG for reproducibility (do NOT use global np.random)
        self.rng = np.random.default_rng(int(getattr(cfg, "seed", 0)))

        # Step 1: enforce gamma consistency between PPO and potential-based shaping
        self._sync_env_gamma()

    def _sync_env_gamma(self) -> None:
        """
        Single source of truth: PPOConfig.gamma.

        If the environment uses potential-based shaping, it must use the same gamma
        as PPO. We override env.gamma to match cfg.gamma to avoid silent mismatch.
        """
        reward_mode = getattr(self.env, "reward_mode", "terminal")
        if reward_mode != "potential":
            return

        # If env has gamma, force it; if it doesn't, this will fail loudly.
        if not hasattr(self.env, "gamma"):
            raise AttributeError(
                "Env reward_mode is 'potential' but env has no attribute 'gamma'. "
                "Add env.gamma or disable potential shaping."
            )

        env_gamma = float(getattr(self.env, "gamma"))
        ppo_gamma = float(self.cfg.gamma)

        if abs(env_gamma - ppo_gamma) > 1e-12:
            # Override to prevent silent mismatch.
            self.env.gamma = ppo_gamma

        # Optional strict mode: uncomment to hard-fail instead of overriding.
        # assert abs(float(self.env.gamma) - ppo_gamma) <= 1e-12, \
        #     f"Gamma mismatch: env.gamma={self.env.gamma} vs cfg.gamma={ppo_gamma}"

    @torch.no_grad()
    def _act(
        self,
        mol: Chem.Mol,
        actions: Sequence[ActionInstance],
        mask_np: np.ndarray,
        *,
        greedy: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Returns: (act_idx, logp, value)
        If greedy=True, selects argmax action among allowed actions (deterministic evaluation).
        """
        mask = torch.tensor(mask_np, dtype=torch.bool, device=self.device)
        logits = self.model.policy_logits_single(mol, actions, mask)
        value = self.model.value_single(mol)

        if greedy:
            ml = masked_logits(logits, mask)
            idx = torch.argmax(ml)
            dist = torch.distributions.Categorical(logits=ml)
            logp = dist.log_prob(idx)
            return int(idx.item()), float(logp.item()), float(value.item())

        idx, logp = masked_categorical_sample(logits.unsqueeze(0), mask.unsqueeze(0))
        return int(idx.item()), float(logp.item()), float(value.item())

    def collect_rollout(self, lead_smiles: str) -> List[Transition]:
        """
        Collects a batch of transitions from cfg.rollout_episodes episodes.
        Trajectory is flattened; episode boundaries are marked with done=True.
        """
        # Re-sync in case env was changed outside trainer
        self._sync_env_gamma()

        traj: List[Transition] = []

        for _ in range(self.cfg.rollout_episodes):
            self.env.reset(lead_smiles)

            while not self.env.done:
                mol = self.env.state.mol
                actions, mask_np = self.env.available_actions()
                assert mask_np.any(), "No allowed actions; action space broken."

                aidx, logp_old, v_old = self._act(mol, actions, mask_np, greedy=False)
                res = self.env.step(aidx)

                traj.append(
                    Transition(
                        mol=Chem.Mol(mol),
                        actions=list(actions),
                        mask=torch.tensor(mask_np, dtype=torch.bool),  # store on CPU
                        act_idx=aidx,
                        logp_old=logp_old,
                        value_old=v_old,
                        reward=float(res.reward),
                        done=bool(res.done),
                    )
                )

        return traj

    def _compute_gae(
        self, traj: List[Transition]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          act_t: [T] long
          ret_t: [T] float
          adv_t: [T] float

        GAE computed across the flattened trajectory assuming episode boundaries at done=True.
        Rewards are per-step from env; may be terminal-only or shaped depending on env config.
        """
        T = len(traj)
        rewards = np.array([t.reward for t in traj], dtype=np.float32)
        dones = np.array([t.done for t in traj], dtype=np.float32)
        values = np.array([t.value_old for t in traj], dtype=np.float32)

        adv = np.zeros((T,), dtype=np.float32)

        last_gae = 0.0
        next_value = 0.0

        for i in reversed(range(T)):
            if dones[i] == 1.0:
                next_value = 0.0
                last_gae = 0.0

            delta = (
                rewards[i] + self.cfg.gamma * next_value * (1.0 - dones[i]) - values[i]
            )
            last_gae = (
                delta + self.cfg.gamma * self.cfg.lam * (1.0 - dones[i]) * last_gae
            )
            adv[i] = last_gae
            next_value = values[i]

        ret = adv + values

        # advantage normalization
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        ret_t = torch.tensor(ret, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(
            [t.act_idx for t in traj], dtype=torch.long, device=self.device
        )
        return act_t, ret_t, adv_t

    def update(self, traj: List[Transition]) -> dict:
        # Re-sync in case env was changed outside trainer (esp. reward_mode/gamma)
        self._sync_env_gamma()

        act_t, ret_t, adv_t = self._compute_gae(traj)
        logp_old = torch.tensor(
            [t.logp_old for t in traj], dtype=torch.float32, device=self.device
        )

        T = len(traj)
        idxs = np.arange(T)

        stats = {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0}

        for _ in range(self.cfg.update_epochs):
            self.rng.shuffle(idxs)
            for start in range(0, T, self.cfg.minibatch_size):
                mb = idxs[start : start + self.cfg.minibatch_size]
                if len(mb) == 0:
                    continue

                mols = [traj[i].mol for i in mb]
                actions_list = [traj[i].actions for i in mb]
                masks_list = [traj[i].mask.to(torch.bool) for i in mb]

                logits, mask_pad, values = self.model.forward_batch_padded(
                    mols=mols,
                    actions_list=actions_list,
                    masks_list=masks_list,
                    device=self.device,
                )

                actions_mb = act_t[mb]
                adv_mb = adv_t[mb]
                ret_mb = ret_t[mb]
                logp_old_mb = logp_old[mb]

                logp = masked_logprob(logits, mask_pad, actions_mb)
                ent = masked_entropy(logits, mask_pad).mean()

                ratio = torch.exp(logp - logp_old_mb)
                clip = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio
                )
                loss_pi = -(torch.min(ratio * adv_mb, clip * adv_mb)).mean()

                loss_v = F.mse_loss(values, ret_mb)

                loss = loss_pi + self.cfg.vf_coef * loss_v - self.cfg.ent_coef * ent

                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite PPO loss encountered.")

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
                self.opt.step()

                stats["loss_pi"] += float(loss_pi.item())
                stats["loss_v"] += float(loss_v.item())
                stats["entropy"] += float(ent.item())

        denom = max(1, self.cfg.update_epochs * (T // self.cfg.minibatch_size + 1))
        for k in stats:
            stats[k] /= denom
        return stats

    @torch.no_grad()
    def evaluate(
        self,
        lead_smiles: str,
        episodes: int = 64,
        *,
        greedy: bool = True,
    ) -> dict:
        """
        Evaluation should report objective score, not just terminal reward.
        Returns a dict with:
          - mean_final_score
          - mean_return (sum of per-step rewards)
        """
        # Re-sync in case env was changed outside trainer
        self._sync_env_gamma()

        final_scores: List[float] = []
        returns: List[float] = []

        for _ in range(episodes):
            self.env.reset(lead_smiles)
            ep_return = 0.0

            while not self.env.done:
                mol = self.env.state.mol
                actions, mask_np = self.env.available_actions()
                assert mask_np.any(), "No allowed actions; action space broken."

                aidx, _logp, _v = self._act(mol, actions, mask_np, greedy=greedy)
                res = self.env.step(aidx)
                ep_return += float(res.reward)

            score = float(self.env.state.info.get("_score", 0.0))
            final_scores.append(score)
            returns.append(ep_return)

        return {
            "mean_final_score": float(np.mean(final_scores)) if final_scores else 0.0,
            "mean_return": float(np.mean(returns)) if returns else 0.0,
        }
