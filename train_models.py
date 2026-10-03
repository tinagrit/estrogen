from __future__ import annotations

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from models import ARTIFACT_DIR, AFFINITY_PATH, TOXICITY_PATH, featurize_smiles, train_clintox_model


def train_affinity_model() -> RandomForestClassifier:
    from chembl_webresource_client.new_client import new_client

    activities = new_client.activity.filter(
        target_chembl_id="CHEMBL206",
        standard_type__in=["IC50", "Ki", "Kd"],
        pchembl_value__isnull=False,
    ).only(["canonical_smiles", "pchembl_value"])
    features: list[np.ndarray] = []
    labels: list[int] = []
    for activity in activities:
        smiles = activity.get("canonical_smiles")
        try:
            pchembl = float(activity["pchembl_value"])
            if smiles:
                features.append(featurize_smiles(smiles))
                labels.append(int(pchembl >= 6.0))
        except (KeyError, TypeError, ValueError):
            continue
    if len(features) < 20 or len(set(labels)) < 2:
        raise RuntimeError("ChEMBL206 returned too few usable active/inactive records.")
    model = RandomForestClassifier(
        n_estimators=400, class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    model.fit(np.vstack(features), np.asarray(labels))
    return model


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    toxicity = train_clintox_model()
    joblib.dump(toxicity, TOXICITY_PATH)
    affinity = train_affinity_model()
    joblib.dump(affinity, AFFINITY_PATH)
    print(f"Saved affinity and ClinTox models to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()