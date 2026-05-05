from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rdkit.Chem import GetPeriodicTable

from ..core.rdkit_utils import mol_from_smiles
from .props import compute_props, delta_props
from .schema import EpisodeRecord, StepRecord

_PT = GetPeriodicTable()


def _sym(z: int) -> str:
    try:
        return _PT.GetElementSymbol(int(z))
    except Exception:
        return f"Z{z}"


def _step_detail(operator: str, site: tuple[int, ...], template, payload: dict) -> str:
    if operator == "AtomMutation":
        f = payload.get("from")
        t = payload.get("to")
        if f is not None and t is not None:
            return f"{_sym(f)}→{_sym(t)} at atom {site[0] if site else '?'}"
        return f"Atom mutation at {site}"

    if operator == "FunctionalGroupSwap":
        f = payload.get("from")
        t = payload.get("to")
        if f is not None and t is not None:
            return f"FG swap {_sym(f)}→{_sym(t)} at atom {site[0] if site else '?'}"
        return f"FG swap at {site}"

    if operator == "PruneTerminal":
        rz = payload.get("removed_z")
        nidx = payload.get("neighbor_idx")
        nz = payload.get("neighbor_z")
        ridx = payload.get("removed_atom_idx", site[0] if site else "?")
        if rz is not None:
            return f"Removed terminal {_sym(rz)} (atom {ridx}) attached to {_sym(nz)} (atom {nidx})"
        return f"Pruned terminal atom at {site}"

    if operator == "AddSubstituent":
        frag = payload.get("frag_smiles", "")
        az = payload.get("attach_z")
        aidx = payload.get("attach_atom_idx", site[0] if site else "?")
        if az is not None:
            return f"Attach {template} ({frag}) to {_sym(az)} (atom {aidx})"
        return f"Attach {template} ({frag}) at {site}"

    if operator == "Terminate":
        return "Terminate"

    return f"{operator} at {site}"


@dataclass
class SARLogger:
    """
    Collects episode trajectories and produces EpisodeRecord objects.
    Assumes GraphEnvironment stores trajectory entries in state.info["trajectory"].
    """

    records: List[EpisodeRecord] = None
    _episode_counter: int = 0

    def __post_init__(self) -> None:
        if self.records is None:
            self.records = []

    def start_episode(self) -> int:
        eid = self._episode_counter
        self._episode_counter += 1
        return eid

    def log_episode(
        self,
        *,
        episode_id: int,
        lead_smiles: str,
        final_smiles: str,
        lead_score: float = 0.0,
        final_score: float,
        trajectory: list[dict],
    ) -> EpisodeRecord:
        """
        Convert an environment trajectory into a structured EpisodeRecord.

        Notes:
        - lead_score/final_score should both be in [0,1] for typical scorer usage.
        - trajectory is expected to be env.state.info["trajectory"] list of dicts.
        """
        lead_mol = mol_from_smiles(lead_smiles, sanitize=True)
        final_mol = mol_from_smiles(final_smiles, sanitize=True)

        lead_p = compute_props(lead_mol)
        final_p = compute_props(final_mol)
        d_p = delta_props(lead_p, final_p)

        steps: List[StepRecord] = []
        op_seq: List[str] = []
        site_seq: List[tuple[int, ...]] = []

        for item in trajectory:
            act = item.get("action", {})
            op = str(act.get("operator", ""))
            site = tuple(act.get("site", ()))
            tpl = act.get("template", None)
            payload = dict(act.get("payload", {}) or {})

            detail = _step_detail(op, site, tpl, payload)

            steps.append(
                StepRecord(
                    t=int(item.get("t", -1)),
                    operator=op,
                    site=site,
                    template=tpl,
                    payload=payload,
                    detail=detail,
                    smiles_before=str(item.get("smiles_before", "")),
                    smiles_after=str(item.get("smiles_after", "")),
                )
            )
            op_seq.append(op)
            site_seq.append(site)

        delta_score = float(final_score) - float(lead_score)

        rec = EpisodeRecord(
            episode_id=episode_id,
            lead_smiles=lead_smiles,
            final_smiles=final_smiles,
            lead_score=float(lead_score),
            final_score=float(final_score),
            delta_score=float(delta_score),
            steps=steps,
            lead_props=lead_p,
            final_props=final_p,
            delta_props=d_p,
            operator_sequence=op_seq,
            site_sequence=site_seq,
        )

        self.records.append(rec)
        return rec
