from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ACTIVITY_PATH = ARTIFACT_DIR / "affinity.joblib"
# Backward-compatible alias for the existing artifact filename and earlier imports.
AFFINITY_PATH = ACTIVITY_PATH
TOXICITY_PATH = ARTIFACT_DIR / "toxicity.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=1024,
)

def featurize_smiles(smiles: str) -> np.ndarray:
    """Return a 1024-bit radius-2 Morgan fingerprint for a valid SMILES."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fingerprint = MORGAN_GENERATOR.GetFingerprint(molecule)
    features = np.zeros((1024,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, features)
    return features


def load_clintox_splits() -> dict[str, list[dict[str, Any]]]:
    """Load ClinTox lazily and preserve its official dataset splits."""
    from datasets import load_dataset

    dataset = load_dataset("zpn/clintox")
    result: dict[str, list[dict[str, Any]]] = {}
    for split_name, split in dataset.items():
        rows: list[dict[str, Any]] = []
        for row in split:
            smiles = row.get("smiles") or row.get("SMILES")
            label = row.get("CT_TOX", row.get("target"))
            if smiles and label is not None:
                rows.append({"smiles": str(smiles), "label": int(label)})
        if rows:
            result[str(split_name)] = rows
    if not result:
        raise ValueError("The ClinTox dataset has no rows with smiles and CT_TOX.")
    return result


def load_clintox_dataset() -> list[dict[str, Any]]:
    """Return all labeled ClinTox rows across the available official splits."""
    return [row for rows in load_clintox_splits().values() for row in rows]


def _neutral_estimator() -> DummyClassifier:
    estimator = DummyClassifier(strategy="prior")
    estimator.fit(np.zeros((2, 1024), dtype=np.uint8), np.array([0, 1]))
    return estimator


@lru_cache(maxsize=1)
def load_predictors() -> tuple[Any, Any]:
    """Load trained artifacts, or use neutral placeholders for missing models."""
    # Keep independently trained models available if one data source is offline.
    activity = (
        joblib.load(ACTIVITY_PATH)
        if ACTIVITY_PATH.is_file()
        else _neutral_estimator()
    )
    toxicity = (
        joblib.load(TOXICITY_PATH)
        if TOXICITY_PATH.is_file()
        else _neutral_estimator()
    )
    return activity, toxicity


def predict_scores(smiles: str) -> tuple[float, float]:
    """Return predicted ER-alpha activity and ClinTox toxicity probabilities."""
    features = featurize_smiles(smiles).reshape(1, -1)
    activity_model, toxicity_model = load_predictors()
    activity_index = list(activity_model.classes_).index(1)
    toxicity_index = list(toxicity_model.classes_).index(1)
    activity = float(activity_model.predict_proba(features)[0, activity_index])
    toxicity = float(toxicity_model.predict_proba(features)[0, toxicity_index])
    return activity, toxicity


def train_clintox_model() -> RandomForestClassifier:
    """Fit a toxicity model from ClinTox CT_TOX labels."""
    rows = load_clintox_dataset()
    features: list[np.ndarray] = []
    labels: list[int] = []
    for row in rows:
        try:
            features.append(featurize_smiles(row["smiles"]))
            labels.append(row["label"])
        except (ValueError, TypeError):
            continue
    if len(set(labels)) < 2:
        raise ValueError("ClinTox needs both toxic and non-toxic examples to train.")
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample", random_state=42,
        n_jobs=-1,
    )
    model.fit(np.vstack(features), np.asarray(labels, dtype=np.int64))
    return model
