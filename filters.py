from __future__ import annotations

from dataclasses import asdict, dataclass

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

try:
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    sascorer = None


@dataclass
class MoleculeMetrics:
    smiles: str
    molecular_weight: float
    logp: float
    hbd: int
    hba: int
    sa_score: float

    def as_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


def calculate_sa_score(molecule: Chem.Mol) -> float:
    if sascorer is not None:
        return float(sascorer.calculateScore(molecule))
    atom_count = molecule.GetNumHeavyAtoms()
    ring_count = rdMolDescriptors.CalcNumRings(molecule)
    stereo_count = len(Chem.FindMolChiralCenters(molecule, includeUnassigned=True))
    score = 1.0 + atom_count / 18.0 + ring_count * 0.28 + stereo_count * 0.35
    return min(10.0, max(1.0, score))


def analyze_smiles(smiles: str) -> MoleculeMetrics | None:
    try:
        molecule = Chem.MolFromSmiles(smiles, sanitize=False)
        if molecule is None:
            return None
        Chem.SanitizeMol(molecule)
        canonical = Chem.MolToSmiles(molecule, canonical=True)
        return MoleculeMetrics(
            smiles=canonical,
            molecular_weight=float(Descriptors.MolWt(molecule)),
            logp=float(Crippen.MolLogP(molecule)),
            hbd=int(Lipinski.NumHDonors(molecule)),
            hba=int(Lipinski.NumHAcceptors(molecule)),
            sa_score=calculate_sa_score(molecule),
        )
    except (ValueError, TypeError, RuntimeError):
        return None


def passes_filters(metrics: MoleculeMetrics) -> bool:
    return (
        metrics.molecular_weight <= 500
        and metrics.logp <= 5.0
        and metrics.hbd <= 5
        and metrics.hba <= 10
        and metrics.sa_score <= 6.0
    )


def filter_candidates(smiles_list: list[str]) -> list[MoleculeMetrics]:
    results: list[MoleculeMetrics] = []
    seen: set[str] = set()
    for smiles in smiles_list:
        metrics = analyze_smiles(smiles)
        if metrics and metrics.smiles not in seen and passes_filters(metrics):
            seen.add(metrics.smiles)
            results.append(metrics)
    return results