from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from rdkit import Chem

from ..actions import ActionInstance
from .action_vocab import ActionVocab
from .featurizers import mol_to_graph_tensors


def masked_logits(
    logits: torch.Tensor, mask: torch.Tensor, neg_inf: float = -1e9
) -> torch.Tensor:
    neg = torch.tensor(neg_inf, device=logits.device, dtype=logits.dtype)
    return torch.where(mask, logits, neg)


def masked_categorical_sample(
    logits: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    logits: [B, A]
    mask:   [B, A] bool
    Returns (idx [B], logp [B])
    """
    ml = masked_logits(logits, mask)
    dist = torch.distributions.Categorical(logits=ml)
    idx = dist.sample()
    logp = dist.log_prob(idx)
    return idx, logp


def masked_logprob(
    logits: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """
    logits:  [B, A]
    mask:    [B, A] bool
    actions: [B] long
    """
    ml = masked_logits(logits, mask)
    dist = torch.distributions.Categorical(logits=ml)
    return dist.log_prob(actions)


def masked_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ml = masked_logits(logits, mask)
    dist = torch.distributions.Categorical(logits=ml)
    return dist.entropy()


class SimpleMPNN(nn.Module):
    def __init__(
        self, atom_dim: int, bond_dim: int, hidden_dim: int = 128, steps: int = 3
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.steps = steps

        self.atom_proj = nn.Linear(atom_dim, hidden_dim)
        self.edge_proj = nn.Linear(bond_dim, hidden_dim)

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          node_h:  [N, D]
          graph_g: [D] (mean pooled)
        """
        h = self.atom_proj(x)  # [N, D]

        if edge_index.numel() == 0:
            g = h.mean(dim=0)
            return h, g

        src, dst = edge_index[0], edge_index[1]  # [E2]
        e = self.edge_proj(edge_attr)  # [E2, D]

        for _ in range(self.steps):
            m_in = torch.cat([h[src], e], dim=-1)  # [E2, 2D]
            m = self.msg_mlp(m_in)  # [E2, D]

            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)

            h = self.gru(agg, h)

        g = h.mean(dim=0)
        return h, g


class MPNNPolicy(nn.Module):
    """
    Variable-action policy + value network.

    - Encodes molecule into node embeddings and graph embedding using an MPNN.
    - Policy scores each ActionInstance with an MLP over:
        [graph embedding, site embedding, operator embedding, template embedding]
    - Value is computed from the graph embedding.

    Key properties:
    - Uses ActionVocab for stable operator/template IDs (no collisions, reproducible).
    - Applies action masking inside the policy (safe-by-default).
    """

    def __init__(
        self,
        atom_feat_dim: int,
        bond_feat_dim: int,
        vocab: ActionVocab,
        *,
        hidden_dim: int = 128,
        mp_steps: int = 3,
        emb_dim: int = 32,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab = vocab

        self.encoder = SimpleMPNN(
            atom_dim=atom_feat_dim,
            bond_dim=bond_feat_dim,
            hidden_dim=hidden_dim,
            steps=mp_steps,
        )

        self.op_emb = nn.Embedding(vocab.num_ops, emb_dim)
        self.tpl_emb = nn.Embedding(vocab.num_tpl, emb_dim)

        in_dim = hidden_dim + hidden_dim + emb_dim + emb_dim
        self.policy_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _site_to_atom_index(site) -> Optional[int]:
        if site is None:
            return None
        if isinstance(site, int):
            return site
        if isinstance(site, (tuple, list)) and len(site) > 0:
            try:
                return int(site[0])
            except Exception:
                return None
        return None

    def encode_single(
        self, mol: Chem.Mol, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = device or next(self.parameters()).device
        x, edge_index, edge_attr = mol_to_graph_tensors(mol, dev)
        node_h, graph_g = self.encoder(x, edge_index, edge_attr)
        return node_h, graph_g

    def _action_features(
        self,
        node_h: torch.Tensor,  # [N, D]
        graph_g: torch.Tensor,  # [D]
        action: ActionInstance,
    ) -> torch.Tensor:
        # Site embedding
        idx = self._site_to_atom_index(action.site)
        if idx is not None and 0 <= idx < node_h.size(0):
            site_h = node_h[idx]
        else:
            site_h = torch.zeros_like(graph_g)

        # Stable vocab IDs (no hashing)
        op_id = self.vocab.op_id(action.operator)
        tpl_id = self.vocab.tpl_id(action.operator, action.template)

        dev = node_h.device
        op_e = self.op_emb(torch.tensor(op_id, device=dev, dtype=torch.long))
        tpl_e = self.tpl_emb(torch.tensor(tpl_id, device=dev, dtype=torch.long))

        return torch.cat([graph_g, site_h, op_e, tpl_e], dim=-1)

    def policy_logits_from_encoding(
        self,
        node_h: torch.Tensor,
        graph_g: torch.Tensor,
        actions: Sequence[ActionInstance],
        mask: torch.Tensor,  # [A] bool
    ) -> torch.Tensor:
        if len(actions) == 0:
            return torch.zeros((0,), dtype=torch.float32, device=graph_g.device)

        feats = torch.stack(
            [self._action_features(node_h, graph_g, a) for a in actions], dim=0
        )
        logits = self.policy_head(feats).squeeze(-1)

        if mask.shape != logits.shape:
            raise ValueError(f"mask shape {mask.shape} != logits shape {logits.shape}")

        # Safe-by-default: apply the mask here
        return masked_logits(logits, mask)

    def value_from_encoding(self, graph_g: torch.Tensor) -> torch.Tensor:
        return self.value_head(graph_g).squeeze(-1)

    def policy_logits_single(
        self,
        mol: Chem.Mol,
        actions: Sequence[ActionInstance],
        mask: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        dev = device or next(self.parameters()).device
        node_h, graph_g = self.encode_single(mol, dev)
        return self.policy_logits_from_encoding(node_h, graph_g, actions, mask)

    def value_single(
        self, mol: Chem.Mol, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        dev = device or next(self.parameters()).device
        _node_h, graph_g = self.encode_single(mol, dev)
        return self.value_from_encoding(graph_g)

    def forward_batch_padded(
        self,
        mols: Sequence[Chem.Mol],
        actions_list: Sequence[Sequence[ActionInstance]],
        masks_list: Sequence[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = device or next(self.parameters()).device
        B = len(mols)
        if not (len(actions_list) == len(masks_list) == B):
            raise ValueError("Batch lists must have same length.")
        Amax = max((len(a) for a in actions_list), default=0)

        logits_pad = torch.zeros((B, Amax), dtype=torch.float32, device=dev)
        mask_pad = torch.zeros((B, Amax), dtype=torch.bool, device=dev)
        values = torch.zeros((B,), dtype=torch.float32, device=dev)

        for i in range(B):
            acts = list(actions_list[i])
            m = masks_list[i].to(dev)

            node_h, graph_g = self.encode_single(mols[i], dev)
            values[i] = self.value_from_encoding(graph_g)

            if len(acts) == 0:
                continue
            if m.shape[0] != len(acts):
                raise ValueError("Mask length must equal action count.")

            li = self.policy_logits_from_encoding(node_h, graph_g, acts, m)
            logits_pad[i, : len(acts)] = li
            mask_pad[i, : len(acts)] = m

        return logits_pad, mask_pad, values
