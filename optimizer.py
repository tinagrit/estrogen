from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from filters import analyze_smiles, passes_filters
from models import predict_scores


TRANSFORMATIONS = (
    ("Phenol to fluoro", "[c:1][OH:2]>>[c:1][F:2]"),
    ("Aryl fluoro to phenol", "[c:1][F:2]>>[c:1][OH:2]"),
    ("Aryl chloro to fluoro", "[c:1][Cl:2]>>[c:1][F:2]"),
    ("Aryl bromo to fluoro", "[c:1][Br:2]>>[c:1][F:2]"),
    ("Aryl chloro to phenol", "[c:1][Cl:2]>>[c:1][OH:2]"),
    ("Aryl bromo to phenol", "[c:1][Br:2]>>[c:1][OH:2]"),
    ("Aryl methoxy demethylation", "[c:1][O:2][CH3:3]>>[c:1][OH:2].[CH4:3]"),
    ("Ether demethylation", "[O;X2:1][CH3:2]>>[OH:1].[CH4:2]"),
    ("N-demethylation", "[N;X3;+0:1][CH3:2]>>[NH:1].[CH4:2]"),
    ("Tertiary amine hydroxymethylation", "[N;X3;+0:1]([CH3:2])[CH3:3]>>[N:1]([CH3:2])[CH2:3]O"),
    ("Aryl methyl to phenol", "[c:1][CH3:2]>>[c:1][OH:2]"),
    ("Aryl methyl to fluoro", "[c:1][CH3:2]>>[c:1][F:2]"),
    ("Aromatic carbon to nitrogen", "[cH:1]>>[n:1]"),
    ("Aromatic nitrogen to carbon", "[nH0;+0:1]>>[cH:1]"),
    ("Carboxylic acid to amide", "[C:1](=[O:2])[OH:3]>>[C:1](=[O:2])[NH2:3]"),
    ("Primary amide to acid", "[C:1](=[O:2])[NH2:3]>>[C:1](=[O:2])[OH:3]"),
    ("Methyl ester hydrolysis", "[C:1](=[O:2])[O:3][CH3:4]>>[C:1](=[O:2])[OH:3].[CH4:4]"),
    ("Alcohol to fluoro", "[C;X4:1][OH:2]>>[C:1][F:2]"),
    ("Alkyl fluoro to alcohol", "[C;X4:1][F:2]>>[C:1][OH:2]"),
    ("Saturated ring carbon to nitrogen", "[CH2;R:1]>>[NH;R:1]"),
    ("Saturated ring nitrogen to carbon", "[NH;R:1]>>[CH2;R:1]"),
    ("Terminal methyl to alcohol", "[CH3:1][C:2]>>[OH:1][C:2]"),
    ("Terminal methyl to fluoro", "[CH3:1][C:2]>>[F:1][C:2]"),
)


def _analog_smiles(molecule: Chem.Mol) -> dict[str, set[str]]:
    """Return canonical analogs and the transformations that produced them."""
    reference = Chem.MolToSmiles(molecule, canonical=True)
    analogs: dict[str, set[str]] = {}
    for name, pattern in TRANSFORMATIONS:
        try:
            reaction = rdChemReactions.ReactionFromSmarts(pattern)
            for product_tuple in reaction.RunReactants((molecule,)):
                product = product_tuple[0]
                Chem.SanitizeMol(product)
                smiles = Chem.MolToSmiles(product, canonical=True)
                if smiles != reference:
                    analogs.setdefault(smiles, set()).add(name)
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
        "passes_filters": passes_filters(baseline_metrics),
        "affinity": baseline_affinity,
        "toxicity": baseline_toxicity,
    }

    molecule = Chem.MolFromSmiles(baseline_metrics.smiles)
    if molecule is None:
        return baseline, []
    candidates: list[dict] = []
    for candidate_smiles, transformations in _analog_smiles(molecule).items():
        metrics = analyze_smiles(candidate_smiles)
        if metrics is None:
            continue
        try:
            affinity, toxicity = predict_scores(metrics.smiles)
        except (ValueError, RuntimeError):
            continue
        row = {
            **metrics.as_dict(),
            "transformations": ", ".join(sorted(transformations)),
            "passes_filters": passes_filters(metrics),
            "affinity": affinity,
            "toxicity": toxicity,
            "delta_affinity": affinity - baseline_affinity,
            "delta_logp": metrics.logp - baseline_metrics.logp,
            "delta_toxicity": toxicity - baseline_toxicity,
        }
        if objective == "Decrease LogP":
            objective_delta = row["delta_logp"]
            improves = objective_delta < 0
        elif objective == "Lower Toxicity Risk":
            objective_delta = row["delta_toxicity"]
            improves = objective_delta < 0
        else:
            objective_delta = row["delta_affinity"]
            improves = objective_delta > 0
        row["objective_delta"] = objective_delta
        row["improves_objective"] = improves
        candidates.append(row)

    sort_key = {
        "Decrease LogP": "delta_logp",
        "Lower Toxicity Risk": "delta_toxicity",
        "Increase ERα Affinity": "delta_affinity",
    }.get(objective, "delta_affinity")
    reverse = objective == "Increase ERα Affinity"
    candidates.sort(key=lambda row: row[sort_key], reverse=reverse)
    return baseline, candidates[:limit]
