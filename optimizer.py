from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from filters import analyze_smiles
from models import predict_scores


TRANSFORMATIONS = (
    "[c:1][OH:2]>>[c:1]F",
    "[c:1][F:2]>>[c:1][OH]",
    "[c:1][O:2][CH3:3]>>[c:1][CH3:3]",
    "[N:1]([CH3:2])[CH3:3]>>[NH:1][CH3:2]",
    "[N:1]([CH3:2])[CH3:3]>>[N:1]([CH3:2])CO",
    "[c:1][CH3:2]>>[c:1]O",
)


def _analog_smiles(molecule: Chem.Mol) -> set[str]:
    analogs: set[str] = set()
    for pattern in TRANSFORMATIONS:
        try:
            reaction = rdChemReactions.ReactionFromSmarts(pattern)
            for product_tuple in reaction.RunReactants((molecule,)):
                product = product_tuple[0]
                Chem.SanitizeMol(product)
                analogs.add(Chem.MolToSmiles(product, canonical=True))
        except (ValueError, RuntimeError):
            continue
    return analogs


def optimize_smiles(smiles: str, objective: str, limit: int = 15) -> tuple[dict, list[dict]]:
    baseline_metrics = analyze_smiles(smiles)
    if baseline_metrics is None:
        raise ValueError("Enter a valid SMILES string.")
    baseline_affinity, baseline_toxicity = predict_scores(baseline_metrics.smiles)
    baseline = {
        **baseline_metrics.as_dict(),
        "affinity": baseline_affinity,
        "toxicity": baseline_toxicity,
    }

    molecule = Chem.MolFromSmiles(baseline_metrics.smiles)
    if molecule is None:
        return baseline, []
    ranked: list[dict] = []
    for candidate_smiles in _analog_smiles(molecule):
        metrics = analyze_smiles(candidate_smiles)
        if metrics is None:
            continue
        try:
            affinity, toxicity = predict_scores(metrics.smiles)
        except (ValueError, RuntimeError):
            continue
        row = {
            **metrics.as_dict(),
            "affinity": affinity,
            "toxicity": toxicity,
            "delta_affinity": affinity - baseline_affinity,
            "delta_logp": metrics.logp - baseline_metrics.logp,
            "delta_toxicity": toxicity - baseline_toxicity,
        }
        if objective == "Decrease LogP" and row["delta_logp"] < 0:
            ranked.append(row)
        elif objective == "Lower Toxicity Risk" and row["delta_toxicity"] < 0:
            ranked.append(row)
        elif objective == "Increase ERα Affinity" and row["delta_affinity"] > 0:
            ranked.append(row)

    sort_key = {
        "Decrease LogP": "delta_logp",
        "Lower Toxicity Risk": "delta_toxicity",
        "Increase ERα Affinity": "delta_affinity",
    }.get(objective, "delta_affinity")
    reverse = objective == "Increase ERα Affinity"
    ranked.sort(key=lambda row: row[sort_key], reverse=reverse)
    return baseline, ranked[:limit]