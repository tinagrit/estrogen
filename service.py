from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from threading import Lock
from typing import Any

from rdkit import Chem
from rdkit.Chem import Draw

from filters import analyze_smiles, filter_candidates
from generator import generate_smiles
from models import predict_scores
from optimizer import optimize_smiles


_GENERATION_LOCK = Lock()


@lru_cache(maxsize=4096)
def score_molecule(smiles: str) -> dict[str, str | float | int] | None:
    """Validate and score a molecule for API and UI consumers."""
    metrics = analyze_smiles(smiles)
    if metrics is None:
        return None
    activity, toxicity = predict_scores(metrics.smiles)
    return {
        **metrics.as_dict(),
        "predicted_eralpha_activity_probability": activity,
        "predicted_clintox_toxicity_probability": toxicity,
        "predicted_eralpha_activity_percent": round(activity * 100, 1),
        "predicted_clintox_toxicity_percent": round(toxicity * 100, 1),
    }


def generate_candidates(
    prefix: str,
    batch_size: int,
    temperature: float,
    top_k: int,
    max_length: int,
) -> dict[str, Any]:
    """Generate, filter, and score molecules using the existing pipeline."""
    prefix = prefix.strip()
    if not prefix:
        raise ValueError("A SMILES prefix is required.")
    # A single in-process generator prevents concurrent requests from duplicating
    # the model's peak memory use. Scoring can still run concurrently.
    with _GENERATION_LOCK:
        generated = generate_smiles(
            prefix,
            num_return_sequences=batch_size,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
        )
    feasible = filter_candidates(generated)
    candidates = [
        result
        for candidate in feasible
        if (result := score_molecule(candidate.smiles)) is not None
    ]
    return {
        "input": {
            "prefix": prefix,
            "batch_size": batch_size,
            "temperature": temperature,
            "top_k": top_k,
            "max_length": max_length,
        },
        "counts": {
            "unique_valid_generations": len(generated),
            "passed_rdkit_filters": len(feasible),
            "scored_candidates": len(candidates),
        },
        "candidates": candidates,
    }


def optimize_candidates(
    smiles: str,
    objective: str,
    limit: int = 15,
) -> dict[str, Any]:
    """Generate and rank analogs using the existing optimizer."""
    smiles = smiles.strip()
    if not smiles:
        raise ValueError("A complete SMILES string is required.")
    baseline, analogs = optimize_smiles(smiles, objective, limit=limit)
    improved_count = sum(bool(row.get("improves_objective")) for row in analogs)
    if not analogs:
        message = "No supported structural transformation matched this molecule."
    elif not improved_count:
        message = (
            "Valid analogs were generated, but none improved the selected objective. "
            "The closest candidates are shown."
        )
    else:
        message = (
            f"{improved_count} of {len(analogs)} displayed analogs improve the "
            "selected objective."
        )
    return {
        "input": {"smiles": smiles, "objective": objective, "limit": limit},
        "baseline": baseline,
        "counts": {
            "displayed_analogs": len(analogs),
            "improved_analogs": improved_count,
        },
        "message": message,
        "analogs": analogs,
    }


def molecule_png(smiles: str, size: tuple[int, int] = (560, 360)) -> bytes:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES string.")
    image = Draw.MolToImage(molecule, size=size)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def sdf_export(rows: list[dict[str, Any]]) -> str:
    records: list[str] = []
    for row in rows:
        smiles = str(row.get("smiles", ""))
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        record = Chem.MolToMolBlock(molecule)
        for key, value in row.items():
            record += f">  <{key}>\n{value}\n\n"
        records.append(record + "$$$$\n")
    return "".join(records)
