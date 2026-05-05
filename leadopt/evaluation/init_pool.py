from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


def load_smiles_pool_jsonl(path: str | Path) -> Tuple[List[str], List[float]]:
    """
    Load a beam-search pool JSONL file.
    Each line must contain at least {"smiles": "..."} and optionally {"objective": <float>}.

    Returns (smiles_list, objective_list). If objective missing, uses 0.0.
    """
    p = Path(path)
    smiles: List[str] = []
    scores: List[float] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        s = str(obj["smiles"])
        v = float(obj.get("objective", 0.0))
        smiles.append(s)
        scores.append(v)
    if not smiles:
        raise ValueError(f"No entries loaded from {p}")
    return smiles, scores
