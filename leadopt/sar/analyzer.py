# leadopt/sar/analyzer.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rdkit import Chem

from ..core.rdkit_utils import mol_from_smiles
from ..evaluation.diversity import compute_diversity_metrics
from .conditional_effects import (
    ConditionalEffectsConfig,
    conditional_edit_effects,
    template_context_heterogeneity,
)
from .schema import EpisodeRecord
from .statistics import bootstrap_ci


@dataclass
class SARAnalyzer:
    records: List[EpisodeRecord]

    # ------------------------------------------------------------------
    # DataFrames
    # ------------------------------------------------------------------

    def to_episode_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            lead_score = float(getattr(r, "lead_score", 0.0))
            final_score = float(r.final_score)
            delta_score = float(getattr(r, "delta_score", final_score - lead_score))

            row = {
                "episode_id": r.episode_id,
                "lead_smiles": r.lead_smiles,
                "final_smiles": r.final_smiles,
                "lead_score": lead_score,
                "final_score": final_score,
                "delta_score": delta_score,
                "n_steps": len(r.steps),
                "operator_seq": " > ".join(r.operator_sequence),
            }

            for k, v in r.lead_props.items():
                row[f"lead_{k}"] = v
            for k, v in r.final_props.items():
                row[f"final_{k}"] = v
            for k, v in r.delta_props.items():
                row[k] = v

            rows.append(row)

        return pd.DataFrame(rows)

    def to_step_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            lead_score = float(getattr(r, "lead_score", 0.0))
            final_score = float(r.final_score)
            delta_score = float(getattr(r, "delta_score", final_score - lead_score))

            for s in r.steps:
                rows.append(
                    {
                        "episode_id": r.episode_id,
                        "t": s.t,
                        "operator": s.operator,
                        "site": str(s.site),
                        "template": s.template or "",
                        "detail": getattr(s, "detail", ""),
                        "smiles_before": s.smiles_before,
                        "smiles_after": s.smiles_after,
                        "lead_score": lead_score,
                        "final_score": final_score,
                        "delta_score": delta_score,
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Operator summary with bootstrap CI
    # ------------------------------------------------------------------

    def operator_summary(
        self, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
    ) -> pd.DataFrame:
        rows = []
        for r in self.records:
            lead_score = float(getattr(r, "lead_score", 0.0))
            final_score = float(r.final_score)
            delta_score = float(getattr(r, "delta_score", final_score - lead_score))

            for s in r.steps:
                rows.append(
                    {
                        "episode_id": r.episode_id,
                        "operator": s.operator,
                        "final_score": final_score,
                        "delta_score": delta_score,
                        "delta_MW": float(r.delta_props.get("dMW", 0.0)),
                        "delta_LogP": float(r.delta_props.get("dLogP", 0.0)),
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "operator",
                    "count",
                    "mean_final_score",
                    "final_ci_lo",
                    "final_ci_hi",
                    "mean_delta_score",
                    "delta_ci_lo",
                    "delta_ci_hi",
                    "mean_delta_MW",
                    "mean_delta_LogP",
                ]
            )

        summary_rows = []
        for op, sub in df.groupby("operator"):
            d_ci = bootstrap_ci(
                sub["delta_score"].tolist(), n_boot=n_boot, alpha=alpha, seed=seed
            )
            f_ci = bootstrap_ci(
                sub["final_score"].tolist(), n_boot=n_boot, alpha=alpha, seed=seed + 17
            )

            summary_rows.append(
                {
                    "operator": op,
                    "count": int(len(sub)),
                    "mean_final_score": float(f_ci.mean),
                    "final_ci_lo": float(f_ci.lo),
                    "final_ci_hi": float(f_ci.hi),
                    "mean_delta_score": float(d_ci.mean),
                    "delta_ci_lo": float(d_ci.lo),
                    "delta_ci_hi": float(d_ci.hi),
                    "mean_delta_MW": float(sub["delta_MW"].mean()),
                    "mean_delta_LogP": float(sub["delta_LogP"].mean()),
                }
            )

        g = pd.DataFrame(summary_rows)
        return g.sort_values(
            ["mean_delta_score", "count"], ascending=[False, False]
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Site summary with bootstrap CI
    # ------------------------------------------------------------------

    def site_summary(
        self, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
    ) -> pd.DataFrame:
        rows = []
        for r in self.records:
            lead_score = float(getattr(r, "lead_score", 0.0))
            final_score = float(r.final_score)
            delta_score = float(getattr(r, "delta_score", final_score - lead_score))

            for s in r.steps:
                rows.append(
                    {
                        "episode_id": r.episode_id,
                        "operator": s.operator,
                        "site": str(s.site),
                        "final_score": final_score,
                        "delta_score": delta_score,
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "operator",
                    "site",
                    "count",
                    "mean_final_score",
                    "mean_delta_score",
                    "delta_ci_lo",
                    "delta_ci_hi",
                ]
            )

        summary_rows = []
        for (op, site), sub in df.groupby(["operator", "site"]):
            d_ci = bootstrap_ci(
                sub["delta_score"].tolist(), n_boot=n_boot, alpha=alpha, seed=seed
            )
            summary_rows.append(
                {
                    "operator": op,
                    "site": site,
                    "count": int(len(sub)),
                    "mean_final_score": float(sub["final_score"].mean()),
                    "mean_delta_score": float(d_ci.mean),
                    "delta_ci_lo": float(d_ci.lo),
                    "delta_ci_hi": float(d_ci.hi),
                }
            )

        g = pd.DataFrame(summary_rows)
        return g.sort_values(
            ["mean_delta_score", "count"], ascending=[False, False]
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Edit effect summary with bootstrap CI
    # ------------------------------------------------------------------

    def edit_effect_summary(
        self, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
    ) -> pd.DataFrame:
        """
        Aggregates by (operator, template) and reports association with delta_score.
        Each (operator, template) is counted at most once per episode to avoid long episodes dominating.
        """
        rows = []
        for r in self.records:
            lead_score = float(getattr(r, "lead_score", 0.0))
            final_score = float(r.final_score)
            delta_score = float(getattr(r, "delta_score", final_score - lead_score))

            used = set()
            for s in r.steps:
                key = (s.operator, s.template or "")
                if key in used:
                    continue
                used.add(key)
                rows.append(
                    {
                        "operator": s.operator,
                        "template": s.template or "",
                        "delta_score": delta_score,
                        "final_score": final_score,
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "operator",
                    "template",
                    "n",
                    "mean_delta_score",
                    "delta_ci_lo",
                    "delta_ci_hi",
                    "median_delta_score",
                    "mean_final_score",
                ]
            )

        summary_rows = []
        for (op, tpl), sub in df.groupby(["operator", "template"]):
            d_ci = bootstrap_ci(
                sub["delta_score"].tolist(), n_boot=n_boot, alpha=alpha, seed=seed
            )
            summary_rows.append(
                {
                    "operator": op,
                    "template": tpl,
                    "n": int(len(sub)),
                    "mean_delta_score": float(d_ci.mean),
                    "delta_ci_lo": float(d_ci.lo),
                    "delta_ci_hi": float(d_ci.hi),
                    "median_delta_score": float(sub["delta_score"].median()),
                    "mean_final_score": float(sub["final_score"].mean()),
                }
            )

        g = pd.DataFrame(summary_rows)
        return g.sort_values(
            ["mean_delta_score", "n"], ascending=[False, False]
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Score stats
    # ------------------------------------------------------------------

    def score_stats(self) -> Dict[str, float]:
        scores = [float(r.final_score) for r in self.records]
        if not scores:
            return {"n": 0}
        s = pd.Series(scores, dtype=float)
        return {
            "n": int(s.size),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "min": float(s.min()),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "max": float(s.max()),
        }

    # ------------------------------------------------------------------
    # Report writing (preserves original outputs; adds diversity + CI columns)
    # ------------------------------------------------------------------

    def write_report(
        self, out_dir: str | Path, *, top_k_sdf: int = 50, top_k_examples: int = 5
    ) -> Dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        df_ep = self.to_episode_dataframe()
        df_steps = self.to_step_dataframe()
        df_op = self.operator_summary()
        df_site = self.site_summary()
        df_eff = self.edit_effect_summary()
        stats = self.score_stats()

        # --- NEW (Commit 3): context-conditioned effects ---
        # Uses episode-level delta_score/final_score, conditioned on local edit context keys.
        cfg = ConditionalEffectsConfig(
            radius=2, n_bits=2048, n_boot=2000, alpha=0.05, seed=0, min_n=3
        )
        df_cond = conditional_edit_effects(self.records, cfg=cfg)  # type: ignore[attr-defined]
        df_het = template_context_heterogeneity(df_cond)

        p_ep = out / "episodes.csv"
        p_steps = out / "steps.csv"
        p_op = out / "sar_operator_summary.csv"
        p_site = out / "sar_site_summary.csv"
        p_eff = out / "sar_edit_effects.csv"

        # --- NEW outputs ---
        p_cond = out / "sar_conditional_edit_effects.csv"
        p_het = out / "sar_template_context_heterogeneity.csv"

        p_md = out / "report.md"
        p_sdf = out / "top_molecules.sdf"

        df_ep.to_csv(p_ep, index=False)
        df_steps.to_csv(p_steps, index=False)
        df_op.to_csv(p_op, index=False)
        df_site.to_csv(p_site, index=False)
        df_eff.to_csv(p_eff, index=False)

        # NEW CSVs
        df_cond.to_csv(p_cond, index=False)
        df_het.to_csv(p_het, index=False)

        # Diversity metrics (all finals, vs lead)
        diversity: Dict[str, object] = {}
        if not df_ep.empty:
            mols = [
                mol_from_smiles(str(s), sanitize=True)
                for s in df_ep["final_smiles"].tolist()
            ]
            mols = [m for m in mols if m is not None]
            lead = mol_from_smiles(str(df_ep.iloc[0]["lead_smiles"]), sanitize=True)
            if lead is not None and mols:
                div = compute_diversity_metrics(mols, lead=lead)
                diversity = dict(div.__dict__)

        # Markdown report
        lines: List[str] = []
        lines.append("# SAR Report\n")

        lines.append("## Score statistics\n")
        lines.append("```json\n" + json.dumps(stats, indent=2) + "\n```\n")

        if diversity:
            lines.append("## Diversity metrics\n")
            lines.append("```json\n" + json.dumps(diversity, indent=2) + "\n```\n")

        lines.append("## Operator-level summary (top 20)\n")
        lines.append(df_op.head(20).to_markdown(index=False) + "\n")

        lines.append("## Site-level summary (top 20)\n")
        lines.append(df_site.head(20).to_markdown(index=False) + "\n")

        lines.append("## Edit effects (association with Δscore)\n")
        lines.append(
            "_Interpretation: correlational association across episodes; not causal._\n"
        )
        lines.append(df_eff.head(20).to_markdown(index=False) + "\n")

        # --- NEW (Commit 3): Context-conditioned sections ---
        lines.append(
            "## Context-conditioned edit effects (matched local neighborhood)\n"
        )
        lines.append(
            "_Interpretation: still correlational, but conditioned on a deterministic local context key "
            "(site + neighborhood Morgan bits + size bin) to reduce confounding._\n"
        )
        if df_cond is not None and not df_cond.empty:
            show_cols = [
                "operator",
                "template",
                "site",
                "size_bin",
                "n",
                "mean_delta_score",
                "median_delta_score",
                "delta_ci_lo",
                "delta_ci_hi",
                "mean_final_score",
            ]
            show_cols = [c for c in show_cols if c in df_cond.columns]
            lines.append(df_cond[show_cols].head(20).to_markdown(index=False) + "\n")
            lines.append(f"_Full table written to `{p_cond.name}`._\n")
        else:
            lines.append(
                "_No conditional effects available (insufficient data after min_n filter)._\n"
            )

        lines.append("## Template heterogeneity across contexts\n")
        if df_het is not None and not df_het.empty:
            show_cols = [
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
            show_cols = [c for c in show_cols if c in df_het.columns]
            lines.append(df_het[show_cols].head(20).to_markdown(index=False) + "\n")
            lines.append(f"_Full table written to `{p_het.name}`._\n")
        else:
            lines.append("_No heterogeneity table available._\n")

        # Example trajectories: top episodes by final_score
        lines.append(f"## Example trajectories (top {top_k_examples} episodes)\n")
        if not df_ep.empty and not df_steps.empty:
            top_ids = (
                df_ep.sort_values("final_score", ascending=False)
                .head(top_k_examples)["episode_id"]
                .tolist()
            )
            for eid in top_ids:
                lines.append(f"### Episode {eid}\n")
                sub = df_steps[df_steps["episode_id"] == eid].sort_values("t")
                cols = ["t", "operator", "site", "template", "detail"]
                cols = [c for c in cols if c in sub.columns]
                if len(sub) > 0 and cols:
                    lines.append(sub[cols].to_markdown(index=False) + "\n")
                else:
                    lines.append("_No steps recorded._\n")
        else:
            lines.append("_No trajectories available._\n")

        p_md.write_text("\n".join(lines), encoding="utf-8")

        # SDF of top molecules by score
        if not df_ep.empty:
            df_top = df_ep.sort_values("final_score", ascending=False).head(
                int(top_k_sdf)
            )
            writer = Chem.SDWriter(str(p_sdf))
            for _, row in df_top.iterrows():
                mol = mol_from_smiles(str(row["final_smiles"]), sanitize=True)
                if mol is None:
                    continue
                mol.SetProp("final_score", str(float(row["final_score"])))
                mol.SetProp("lead_score", str(float(row.get("lead_score", 0.0))))
                mol.SetProp("delta_score", str(float(row.get("delta_score", 0.0))))
                mol.SetProp("episode_id", str(int(row["episode_id"])))
                mol.SetProp("operator_seq", str(row.get("operator_seq", "")))

                for col in ["dMW", "dLogP", "dTPSA", "dHeavyAtoms", "dRings", "dRotB"]:
                    if col in row and pd.notna(row[col]):
                        mol.SetProp(col, str(float(row[col])))

                writer.write(mol)
            writer.close()
        else:
            p_sdf.write_text("", encoding="utf-8")

        return {
            "episodes_csv": p_ep,
            "steps_csv": p_steps,
            "operator_csv": p_op,
            "site_csv": p_site,
            "edit_effects_csv": p_eff,
            # NEW
            "conditional_effects_csv": p_cond,
            "template_heterogeneity_csv": p_het,
            "report_md": p_md,
            "top_sdf": p_sdf,
        }
