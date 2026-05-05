from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


@dataclass(frozen=True)
class RuleConfig:
    # Enable standard alert catalogs
    use_pains: bool = False
    use_brenk: bool = False
    use_nih: bool = False

    # Simple “too weird / unstable / reactive” motif bans
    ban_motifs: bool = True

    # Property guardrails (tune per project)
    max_mw: float = 650.0
    max_logp: float = 6.5
    max_rings: int = 8
    max_rot_bonds: int = 12
    max_hbd: int = 6
    max_hba: int = 12
    max_tpsa: float = 200.0

    # Charge / radicals
    max_abs_formal_charge: int = 1
    forbid_radicals: bool = True


# Small, explicit “no-go” motifs: keep conservative, tune later
# (These are examples of genuinely uncommon/reactive motifs in medchem libraries.)
_BANNED_SMARTS: List[Tuple[str, str]] = [
    ("azide", "[N-]=[N+]=N"),
    ("diazo", "[N+]#N"),  # catches diazonium-like; may be aggressive in some contexts
    ("peroxide", "OO"),
    ("acyl_halide", "C(=O)[Cl,Br,I,F]"),
    ("isocyanate", "N=C=O"),
    ("isothiocyanate", "N=C=S"),
    ("diazirine", "C1NN1"),
    ("epoxide", "C1OC1"),
    ("thiirane", "C1SC1"),
    ("strained_3ring_n", "C1NC1"),
    ("N_halogen", "[N;!+]-[F,Cl,Br,I]"),
    ("O_halogen", "[#8]-[F,Cl,Br,I]"),
    ("N_chain", "[N;!a]-[N;!a]-[N;!a]"),
    ("hypervalent_sulfur", "[S;D5,D6,D7]"),
]


@lru_cache(maxsize=8)
def _build_catalog(cfg: RuleConfig) -> Optional[FilterCatalog]:
    if not (cfg.use_pains or cfg.use_brenk or cfg.use_nih):
        return None

    params = FilterCatalogParams()
    if cfg.use_pains:
        # RDKit typically supports PAINS A/B/C catalogs
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    if cfg.use_brenk:
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    if cfg.use_nih:
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)

    return FilterCatalog(params)


@lru_cache(maxsize=1)
def _compiled_banned_smarts() -> List[Tuple[str, Chem.Mol]]:
    out: List[Tuple[str, Chem.Mol]] = []
    for name, sm in _BANNED_SMARTS:
        m = Chem.MolFromSmarts(sm)
        if m is not None:
            out.append((name, m))
    return out


def _num_rings(mol: Chem.Mol) -> int:
    return int(rdMolDescriptors.CalcNumRings(mol))


def check_molecule(mol: Chem.Mol, cfg: RuleConfig) -> Tuple[bool, List[str]]:
    """
    Returns (ok, reasons). Reasons are human-readable strings explaining rejections.
    Assumes mol is already sanitized/valid.
    """
    reasons: List[str] = []

    # radicals
    if cfg.forbid_radicals:
        if any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms()):
            reasons.append("radical_electrons")

    # charge
    total_abs_charge = sum(abs(int(a.GetFormalCharge())) for a in mol.GetAtoms())
    if total_abs_charge > cfg.max_abs_formal_charge:
        reasons.append(f"abs_formal_charge>{cfg.max_abs_formal_charge}")

    # properties (cheap and effective)
    mw = float(Descriptors.MolWt(mol))
    if mw > cfg.max_mw:
        reasons.append(f"MW>{cfg.max_mw:g}")

    logp = float(Crippen.MolLogP(mol))
    if logp > cfg.max_logp:
        reasons.append(f"LogP>{cfg.max_logp:g}")

    rings = _num_rings(mol)
    if rings > cfg.max_rings:
        reasons.append(f"Rings>{cfg.max_rings}")

    rotb = int(Lipinski.NumRotatableBonds(mol))
    if rotb > cfg.max_rot_bonds:
        reasons.append(f"RotB>{cfg.max_rot_bonds}")

    hbd = int(Lipinski.NumHDonors(mol))
    if hbd > cfg.max_hbd:
        reasons.append(f"HBD>{cfg.max_hbd}")

    hba = int(Lipinski.NumHAcceptors(mol))
    if hba > cfg.max_hba:
        reasons.append(f"HBA>{cfg.max_hba}")

    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    if tpsa > cfg.max_tpsa:
        reasons.append(f"TPSA>{cfg.max_tpsa:g}")

    # alert catalogs (PAINS/Brenk/NIH)
    catalog = _build_catalog(cfg)
    if catalog is not None:
        entry = catalog.GetFirstMatch(mol)
        if entry is not None:
            reasons.append(f"alert:{entry.GetDescription()}")

    # explicit banned motifs
    if cfg.ban_motifs:
        for name, patt in _compiled_banned_smarts():
            if mol.HasSubstructMatch(patt):
                reasons.append(f"banned_motif:{name}")

    return (len(reasons) == 0), reasons
