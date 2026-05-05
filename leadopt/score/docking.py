from __future__ import annotations


def example_docking_score(smiles: str) -> float:
    """
    Placeholder deterministic docking-like score (more negative is "better").

    Replace with actual docking (or a DockingScorer class) later.
    """
    return -float(len(smiles))
