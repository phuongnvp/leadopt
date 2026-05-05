# leadopt/config/presets.py
from dataclasses import dataclass
from typing import Optional, Sequence

from leadopt.actions.operators import (
    AddSubstituent,
    AtomMutation,
    AzaScanAromatic,
    BioisostereSwap,
    DeleteSubtree,
    FunctionalGroupSwap,
    LinkerDeleteCH2,
    LinkerInsertCH2,
    PruneTerminal,
    RGroupSwap,
)
from leadopt.core.rules import RuleConfig


@dataclass(frozen=True)
class ActionPreset:
    name: str
    operators: Sequence[object]
    max_steps: int
    rule_config: Optional[RuleConfig] = None


def lead_optimization_preset() -> ActionPreset:
    ops = [
        PruneTerminal(),
        AtomMutation(),
        FunctionalGroupSwap(),
        AddSubstituent(),
        RGroupSwap(max_sidechain_heavy_atoms=12),
        LinkerInsertCH2(),
        LinkerDeleteCH2(),
    ]

    rule_cfg = RuleConfig(
        ban_motifs=True,
        use_pains=False,
        use_brenk=False,
        use_nih=False,
        max_mw=650.0,
        max_logp=6.5,
    )

    return ActionPreset(
        name="lead_optimization",
        operators=ops,
        max_steps=8,
        rule_config=rule_cfg,
    )


def scaffold_hopping_preset() -> ActionPreset:
    ops = [
        PruneTerminal(),
        AtomMutation(),
        FunctionalGroupSwap(),
        AddSubstituent(),
        RGroupSwap(max_sidechain_heavy_atoms=20),
        LinkerInsertCH2(),
        LinkerDeleteCH2(),
        AzaScanAromatic(),
        BioisostereSwap(),  # careful: curated library only
    ]

    rule_cfg = RuleConfig(
        ban_motifs=True,
        use_pains=False,
        use_brenk=False,
        use_nih=False,
        max_mw=650.0,
        max_logp=6.5,
    )
    return ActionPreset(
        name="scaffold_hopping",
        operators=ops,
        max_steps=12,
        rule_config=rule_cfg,
    )


def decomplexification_preset() -> ActionPreset:
    ops = [
        DeleteSubtree(max_deleted_atoms=25),
        PruneTerminal(),
    ]

    rule_cfg = RuleConfig(
        ban_motifs=True,
        use_pains=False,
        use_brenk=False,
        use_nih=False,
        max_mw=650.0,
        max_logp=6.5,
    )
    return ActionPreset(
        name="decomplexification",
        operators=ops,
        max_steps=10,
        rule_config=rule_cfg,
    )
