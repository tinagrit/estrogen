from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from statistics import median
from typing import Any

import joblib
import numpy as np
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from models import (
    ACTIVITY_PATH,
    ARTIFACT_DIR,
    METRICS_PATH,
    TOXICITY_PATH,
    featurize_smiles,
    load_clintox_splits,
)


RANDOM_STATE = 42


def _canonicalize_labeled_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonicalize, deduplicate, and majority-vote repeated binary labels."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        try:
            molecule = Chem.MolFromSmiles(str(row["smiles"]))
            if molecule is None:
                continue
            smiles = Chem.MolToSmiles(molecule, canonical=True)
            grouped[smiles].append(int(row["label"]))
        except (KeyError, TypeError, ValueError):
            continue
    return [
        {"smiles": smiles, "label": int(np.mean(labels) >= 0.5)}
        for smiles, labels in grouped.items()
    ]


def _features_and_labels(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for row in rows:
        try:
            features.append(featurize_smiles(str(row["smiles"])))
            labels.append(int(row["label"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not features or len(set(labels)) < 2:
        raise ValueError("Evaluation requires valid molecules from both label classes.")
    return np.vstack(features), np.asarray(labels, dtype=np.int64)


def _classifier_metrics(
    model: RandomForestClassifier,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    positive_index = list(model.classes_).index(1)
    probability = model.predict_proba(features)[:, positive_index]
    prediction = (probability >= 0.5).astype(np.int64)
    return {
        "samples": int(labels.size),
        "positive_samples": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            labels, prediction, labels=[0, 1]
        ).tolist(),
        "decision_threshold": 0.5,
    }


def _activity_rows() -> list[dict[str, Any]]:
    """Fetch CHEMBL206 and aggregate repeated molecules by median pChEMBL."""
    from chembl_webresource_client.new_client import new_client

    activities = new_client.activity.filter(
        target_chembl_id="CHEMBL206",
        standard_type__in=["IC50", "Ki", "Kd"],
        pchembl_value__isnull=False,
    ).only(["canonical_smiles", "pchembl_value"])
    grouped: dict[str, list[float]] = defaultdict(list)
    for activity in activities:
        try:
            molecule = Chem.MolFromSmiles(str(activity["canonical_smiles"]))
            if molecule is None:
                continue
            smiles = Chem.MolToSmiles(molecule, canonical=True)
            grouped[smiles].append(float(activity["pchembl_value"]))
        except (KeyError, TypeError, ValueError):
            continue
    rows = [
        {
            "smiles": smiles,
            "label": int(median(values) >= 6.0),
            "median_pchembl": float(median(values)),
        }
        for smiles, values in grouped.items()
    ]
    if len(rows) < 20 or len({row["label"] for row in rows}) < 2:
        raise RuntimeError("CHEMBL206 returned too few usable active/inactive molecules.")
    return rows


def train_activity_model() -> tuple[RandomForestClassifier, dict[str, Any]]:
    rows = _activity_rows()
    train_rows, test_rows = train_test_split(
        rows,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=[row["label"] for row in rows],
    )
    train_features, train_labels = _features_and_labels(train_rows)
    test_features, test_labels = _features_and_labels(test_rows)
    model = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train_features, train_labels)
    report = {
        "display_name": "Predicted ERalpha activity probability",
        "dataset": "ChEMBL target CHEMBL206",
        "label": "Active when median pChEMBL >= 6.0",
        "split": "Fixed stratified random molecule split (80% train / 20% test)",
        "train_samples": int(train_labels.size),
        "test": _classifier_metrics(model, test_features, test_labels),
    }
    return model, report


# Backward-compatible name used by earlier setup instructions.
def train_affinity_model() -> RandomForestClassifier:
    model, _ = train_activity_model()
    return model


def train_clintox_model() -> tuple[RandomForestClassifier, dict[str, Any]]:
    splits = load_clintox_splits()
    training_rows = [
        row
        for name in ("train", "validation")
        for row in splits.get(name, [])
    ]
    test_rows = splits.get("test", [])
    if not training_rows or not test_rows:
        all_rows = _canonicalize_labeled_rows(
            [row for rows in splits.values() for row in rows]
        )
        training_rows, test_rows = train_test_split(
            all_rows,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=[row["label"] for row in all_rows],
        )
        split_description = (
            "Fixed stratified random molecule split (80% train / 20% test)"
        )
    else:
        training_rows = _canonicalize_labeled_rows(training_rows)
        training_smiles = {row["smiles"] for row in training_rows}
        test_rows = [
            row
            for row in _canonicalize_labeled_rows(test_rows)
            if row["smiles"] not in training_smiles
        ]
        split_description = (
            "Official ClinTox test split; train and validation used for training"
        )
    train_features, train_labels = _features_and_labels(training_rows)
    test_features, test_labels = _features_and_labels(test_rows)
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train_features, train_labels)
    report = {
        "display_name": "Predicted ClinTox toxicity probability",
        "dataset": "zpn/clintox CT_TOX",
        "label": "CT_TOX binary label",
        "split": split_description,
        "train_samples": int(train_labels.size),
        "test": _classifier_metrics(model, test_features, test_labels),
    }
    return model, report


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("Training and evaluating the ClinTox toxicity classifier...")
    toxicity_model, toxicity_report = train_clintox_model()

    print("Training and evaluating the CHEMBL206 ERalpha activity classifier...")
    activity_model, activity_report = train_activity_model()

    # Do not replace either deployed model unless both training runs succeed.
    joblib.dump(toxicity_model, TOXICITY_PATH)
    joblib.dump(activity_model, ACTIVITY_PATH)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": (
            "Preliminary computational evaluation; metrics do not constitute "
            "experimental or clinical validation."
        ),
        "eralpha_activity": activity_report,
        "clintox_toxicity": toxicity_report,
    }
    METRICS_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"Saved evaluated models and metrics to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
