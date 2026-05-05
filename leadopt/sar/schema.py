from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepRecord:
    t: int
    operator: str
    site: tuple[int, ...]
    template: Optional[str]
    payload: Dict[str, Any]
    detail: str
    smiles_before: str
    smiles_after: str


@dataclass
class EpisodeRecord:
    episode_id: int
    lead_smiles: str
    final_smiles: str
    final_score: float
    lead_score: float = 0.0
    delta_score: float = 0.0
    steps: List[StepRecord] = field(default_factory=list)

    # property dictionaries
    lead_props: Dict[str, float] = field(default_factory=dict)
    final_props: Dict[str, float] = field(default_factory=dict)
    delta_props: Dict[str, float] = field(default_factory=dict)

    # convenience
    operator_sequence: List[str] = field(default_factory=list)
    site_sequence: List[tuple[int, ...]] = field(default_factory=list)
