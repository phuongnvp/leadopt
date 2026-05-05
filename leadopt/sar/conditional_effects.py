"""
leadopt.sar.conditional_effects

Context-conditioned (stratified) edit effect summaries.

We treat each (episode, context-key) as one observation (at most once per episode)
so longer episodes do not dominate.

The primary output is a per-context summary of association between applying an
(operator, template) in a given local context and the episode delta score.

This is still correlational; however, conditioning on a local context reduces a
major source of confounding and yields more defensible interpretability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .context import make_context_key
from .schema import EpisodeRecord
from .statistics import bootstrap_ci


@dataclass
class ConditionalEffectsConfig:
    radius: int = 2
    n_bits: int = 2048
    n_boot: int = 2000
    alpha: float = 0.05
    seed: int = 0
    min_n: int = 3


def conditional_edit_effects(
    records: List[EpisodeRecord],
    *,
    cfg: ConditionalEffectsConfig = ConditionalEffectsConfig(),
) -> pd.DataFrame:
    """Return per-context effect summaries.

    Columns:
      - operator, template, site, radius, n_bits, bit_hash, size_bin
      - context_key (string)
      - n, mean_delta_score, median_delta_score, delta_ci_lo, delta_ci_hi, mean_final_score

    Note: delta_score/final_score are episode-level (terminal reward setting).
    """

    rows: List[Dict[str, object]] = []

    for r in records:
        lead_score = float(getattr(r, "lead_score", 0.0))
        final_score = float(r.final_score)
        delta_score = float(getattr(r, "delta_score", final_score - lead_score))

        seen: set[str] = set()

        for s in r.steps:
            if not getattr(s, "smiles_before", ""):
                continue

            ck = make_context_key(
                smiles_before=str(s.smiles_before),
                site=tuple(s.site),
                operator=str(s.operator),
                template=str(s.template or ""),
                radius=cfg.radius,
                n_bits=cfg.n_bits,
            )
            if ck is None:
                continue

            key_s = ck.as_str()
            if key_s in seen:
                continue
            seen.add(key_s)

            rows.append(
                {
                    "episode_id": int(r.episode_id),
                    "operator": ck.operator,
                    "template": ck.template,
                    "site": str(ck.site),
                    "radius": int(ck.radius),
                    "n_bits": int(ck.n_bits),
                    "bit_hash": ck.bit_hash,
                    "size_bin": ck.size_bin,
                    "context_key": key_s,
                    "delta_score": float(delta_score),
                    "final_score": float(final_score),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "operator",
                "template",
                "site",
                "radius",
                "n_bits",
                "bit_hash",
                "size_bin",
                "context_key",
                "n",
                "mean_delta_score",
                "median_delta_score",
                "delta_ci_lo",
                "delta_ci_hi",
                "mean_final_score",
            ]
        )

    out_rows: List[Dict[str, object]] = []
    for ctx, sub in df.groupby("context_key"):
        if len(sub) < int(cfg.min_n):
            continue

        d_ci = bootstrap_ci(
            sub["delta_score"].tolist(),
            n_boot=cfg.n_boot,
            alpha=cfg.alpha,
            seed=cfg.seed,
        )

        r0 = sub.iloc[0]
        out_rows.append(
            {
                "operator": str(r0["operator"]),
                "template": str(r0["template"]),
                "site": str(r0["site"]),
                "radius": int(r0["radius"]),
                "n_bits": int(r0["n_bits"]),
                "bit_hash": str(r0["bit_hash"]),
                "size_bin": str(r0["size_bin"]),
                "context_key": str(ctx),
                "n": int(len(sub)),
                "mean_delta_score": float(d_ci.mean),
                "median_delta_score": float(sub["delta_score"].median()),
                "delta_ci_lo": float(d_ci.lo),
                "delta_ci_hi": float(d_ci.hi),
                "mean_final_score": float(sub["final_score"].mean()),
            }
        )

    g = pd.DataFrame(out_rows)
    if g.empty:
        return g

    return g.sort_values(
        ["mean_delta_score", "n"], ascending=[False, False]
    ).reset_index(drop=True)


def template_context_heterogeneity(cond_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize heterogeneity across contexts for each (operator, template)."""
    if cond_df is None or cond_df.empty:
        return pd.DataFrame(
            columns=[
                "operator",
                "template",
                "n_contexts",
                "total_n",
                "pooled_mean_delta",
                "context_mean_std",
                "frac_contexts_positive",
                "best_context_mean_delta",
                "worst_context_mean_delta",
            ]
        )

    rows: List[Dict[str, object]] = []
    for (op, tpl), sub in cond_df.groupby(["operator", "template"]):
        n_ctx = int(len(sub))
        total_n = int(sub["n"].sum())

        pooled = float((sub["mean_delta_score"] * sub["n"]).sum() / max(1, total_n))
        std = float(sub["mean_delta_score"].std(ddof=0)) if n_ctx > 1 else 0.0
        frac_pos = float((sub["mean_delta_score"] > 0).mean())

        rows.append(
            {
                "operator": str(op),
                "template": str(tpl),
                "n_contexts": n_ctx,
                "total_n": total_n,
                "pooled_mean_delta": pooled,
                "context_mean_std": std,
                "frac_contexts_positive": frac_pos,
                "best_context_mean_delta": float(sub["mean_delta_score"].max()),
                "worst_context_mean_delta": float(sub["mean_delta_score"].min()),
            }
        )

    g = pd.DataFrame(rows)
    return g.sort_values(
        ["pooled_mean_delta", "n_contexts"], ascending=[False, False]
    ).reset_index(drop=True)
