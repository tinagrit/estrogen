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
AFFINITY_PATH = ARTIFACT_DIR / "affinity.joblib"
TOXICITY_PATH = ARTIFACT_DIR / "toxicity.joblib"

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


def load_clintox_dataset() -> list[dict[str, Any]]:
    """Load ClinTox lazily and return rows with SMILES and CT_TOX labels."""
    from datasets import load_dataset

    dataset = load_dataset("zpn/clintox")
    rows: list[dict[str, Any]] = []
    splits = [dataset["train"]] if "train" in dataset else dataset.values()
    for split in splits:
        for row in split:
            smiles = row.get("smiles") or row.get("SMILES")
            label = row.get("CT_TOX", row.get("target"))
            if smiles and label is not None:
                rows.append({"smiles": str(smiles), "label": int(label)})
    if not rows:
        raise ValueError("The ClinTox dataset has no rows with smiles and CT_TOX.")
    return rows


def _neutral_estimator() -> DummyClassifier:
    estimator = DummyClassifier(strategy="prior")
    estimator.fit(np.zeros((2, 1024), dtype=np.uint8), np.array([0, 1]))
    return estimator


@lru_cache(maxsize=1)
def load_predictors() -> tuple[Any, Any]:
    """Load trained artifacts, or use neutral placeholders for missing models."""
    # Keep independently trained models available if one data source is offline.
    affinity = (
        joblib.load(AFFINITY_PATH)
        if AFFINITY_PATH.is_file()
        else _neutral_estimator()
    )
    toxicity = (
        joblib.load(TOXICITY_PATH)
        if TOXICITY_PATH.is_file()
        else _neutral_estimator()
    )
    return affinity, toxicity


def predict_scores(smiles: str) -> tuple[float, float]:
    """Return ER-alpha activity probability and ClinTox hazard probability."""
    features = featurize_smiles(smiles).reshape(1, -1)
    affinity_model, toxicity_model = load_predictors()
    affinity = float(affinity_model.predict_proba(features)[0, 1])
    toxicity = float(toxicity_model.predict_proba(features)[0, 1])
    return affinity, toxicity


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