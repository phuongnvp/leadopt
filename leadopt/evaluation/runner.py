from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import pandas as pd
from rdkit import Chem

from .diversity import compute_diversity_metrics


@dataclass(frozen=True)
class MethodReport:
    name: str
    episodes_df: pd.DataFrame
    diversity: Dict[str, float]


# A "method" is a callable that can run one episode given an env + lead specification.
# It must return an EpisodeResult-like object with:
#   lead_smiles, final_smiles, lead_score, final_score, trajectory
RunEpisodeFn = Callable[[object, str], object]


def evaluate_methods(
    make_env: Callable[[], object],
    scorer: Callable[
        [Chem.Mol], float
    ],  # kept for interface stability; not strictly needed here
    methods: Dict[str, RunEpisodeFn],
    *,
    lead_smiles: str,
    n_episodes: int = 200,
    top_k: int = 50,
) -> Dict[str, MethodReport]:

    reports: Dict[str, MethodReport] = {}

    lead_mol = Chem.MolFromSmiles(lead_smiles)
    if lead_mol is None:
        raise ValueError(f"Invalid lead_smiles: {lead_smiles!r}")

    for name, run_episode in methods.items():
        rows = []
        finals = []

        for _ in range(int(n_episodes)):
            env = make_env()
            res = run_episode(env, lead_smiles)

            rows.append(
                {
                    "method": name,
                    "lead_smiles": res.lead_smiles,
                    "final_smiles": res.final_smiles,
                    "lead_score": float(res.lead_score),
                    "final_score": float(res.final_score),
                    "delta_score": float(res.final_score) - float(res.lead_score),
                    "n_steps": int(len(res.trajectory)),
                }
            )

            m = Chem.MolFromSmiles(str(res.final_smiles))
            if m is not None:
                finals.append(m)

        df = pd.DataFrame(rows)

        # Diversity on all finals
        div_all = compute_diversity_metrics(finals, lead=lead_mol)

        # Diversity on top-k by final_score
        top = df.sort_values("final_score", ascending=False).head(int(top_k))
        top_mols = [Chem.MolFromSmiles(s) for s in top["final_smiles"].tolist()]
        top_mols = [m for m in top_mols if m is not None]
        div_top = compute_diversity_metrics(top_mols, lead=lead_mol)

        reports[name] = MethodReport(
            name=name,
            episodes_df=df,
            diversity={
                "all_mean_pairwise_tanimoto": float(div_all.mean_pairwise_tanimoto),
                "all_mean_novelty_vs_lead": float(div_all.mean_novelty_vs_lead),
                "all_unique_smiles": float(div_all.unique_smiles),
                "top_mean_pairwise_tanimoto": float(div_top.mean_pairwise_tanimoto),
                "top_mean_novelty_vs_lead": float(div_top.mean_novelty_vs_lead),
                "top_unique_smiles": float(div_top.unique_smiles),
            },
        )

    return reports
