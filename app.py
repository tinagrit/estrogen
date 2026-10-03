from __future__ import annotations

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from filters import analyze_smiles, filter_candidates
from generator import generate_smiles
from models import (
    AFFINITY_PATH,
    TOXICITY_PATH,
    featurize_smiles,
    load_predictors,
)
from optimizer import optimize_smiles


st.set_page_config(page_title="ERα Molecule Studio", page_icon="⚗", layout="wide")
st.title("ERα Molecule Studio")
st.caption("Complete a SMILES prefix or optimize a complete user-provided molecule.")
st.caption("Research prototype only. Scores are computational estimates, not clinical evidence.")


@st.cache_resource
def cached_predictors():
    return load_predictors()


@st.cache_data(show_spinner=False)
def score_cached(smiles: str) -> tuple[float, float]:
    features = featurize_smiles(smiles).reshape(1, -1)
    affinity_model, toxicity_model = cached_predictors()
    affinity_index = list(affinity_model.classes_).index(1)
    toxicity_index = list(toxicity_model.classes_).index(1)
    affinity = float(affinity_model.predict_proba(features)[0, affinity_index])
    toxicity = float(toxicity_model.predict_proba(features)[0, toxicity_index])
    return affinity, toxicity


def metrics_and_scores(smiles: str) -> dict | None:
    metrics = analyze_smiles(smiles)
    if metrics is None:
        return None
    try:
        affinity, toxicity = score_cached(metrics.smiles)
    except Exception as error:
        st.warning(f"Could not score {metrics.smiles}: {error}")
        return None
    return {
        **metrics.as_dict(),
        "affinity_percent": round(affinity * 100, 1),
        "toxicity_percent": round(toxicity * 100, 1),
    }


def molecule_image(smiles: str, size: tuple[int, int] = (300, 220)):
    molecule = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(molecule, size=size) if molecule else None


def sdf_export(rows: list[dict]) -> str:
    records: list[str] = []
    for row in rows:
        molecule = Chem.MolFromSmiles(row["smiles"])
        if molecule is None:
            continue
        record = Chem.MolToMolBlock(molecule)
        for key, value in row.items():
            record += f">  <{key}>\n{value}\n\n"
        records.append(record + "$$$$\n")
    return "".join(records)


def show_results(rows: list[dict], key: str) -> None:
    if not rows:
        st.info("No candidates passed the filters and objective in this run.")
        return
    frame = pd.DataFrame(rows)
    st.dataframe(frame, hide_index=True, use_container_width=True)
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    sdf_bytes = sdf_export(rows).encode("utf-8")
    left, right = st.columns(2)
    left.download_button("Download CSV", csv_bytes, f"{key}.csv", "text/csv")
    right.download_button("Download SDF", sdf_bytes, f"{key}.sdf", "chemical/x-mdl-sdfile")
    for start in range(0, min(len(rows), 15), 3):
        columns = st.columns(3)
        for column, row in zip(columns, rows[start:start + 3]):
            with column:
                image = molecule_image(row["smiles"])
                if image:
                    st.image(image, use_container_width=True)
                st.code(row["smiles"], language=None)


def model_status() -> None:
    missing = []
    if not AFFINITY_PATH.exists():
        missing.append("ERα affinity")
    if not TOXICITY_PATH.exists():
        missing.append("ClinTox toxicity")
    if missing:
        st.warning(
            f"Using an untrained neutral placeholder for {', '.join(missing)}. "
            "Run `python train_models.py` to train the missing model(s)."
        )


tab_generate, tab_optimize = st.tabs(["SMILES Completion", "Lead Optimization"])

with tab_generate:
    st.subheader("Complete a SMILES prefix")
    seed = st.text_input(
        "SMILES prefix to complete",
        placeholder="Enter an incomplete SMILES prefix",
        key="generation_prefix",
    )
    control_a, control_b, control_c, control_d = st.columns(4)
    with control_a:
        batch_size = st.slider("Batch size", min_value=10, max_value=50, value=20, step=5)
    with control_b:
        temperature = st.slider("Temperature", min_value=0.2, max_value=1.5, value=0.8, step=0.1)
    with control_c:
        top_k = st.number_input("Top-k", min_value=1, max_value=200, value=50)
    with control_d:
        max_length = st.number_input("Maximum length", min_value=20, max_value=200, value=80)

    model_status()
    if st.button("Complete and score", type="primary", disabled=not seed.strip()):
        try:
            with st.spinner("Generating molecules and applying chemistry filters..."):
                generated = generate_smiles(
                    seed.strip(), batch_size, temperature, int(top_k), int(max_length)
                )
                feasible = filter_candidates(generated)
                rows = [
                    result for candidate in feasible
                    if (result := metrics_and_scores(candidate.smiles)) is not None
                ]
            st.session_state["generated_rows"] = rows
            st.session_state["generated_prefix"] = seed.strip()
            st.success(f"Kept {len(rows)} candidates from {len(generated)} unique valid generations.")
        except Exception as error:
            st.error(f"Generation failed: {error}")

    if st.session_state.get("generated_prefix") == seed.strip():
        show_results(st.session_state.get("generated_rows", []), "eralpha_candidates")

with tab_optimize:
    st.subheader("Optimize a complete SMILES")
    reference = st.text_input(
        "Complete SMILES to optimize",
        placeholder="Paste a complete SMILES string",
        key="optimization_smiles",
    )
    objective = st.selectbox(
        "Optimization objective",
        ["Decrease LogP", "Lower Toxicity Risk", "Increase ERα Affinity"],
    )
    model_status()
    if st.button("Find analogs", type="primary", disabled=not reference.strip()):
        try:
            baseline, analogs = optimize_smiles(reference, objective, limit=15)
            st.session_state["optimization"] = {
                "smiles": reference.strip(),
                "objective": objective,
                "baseline": baseline,
                "analogs": analogs,
            }
        except Exception as error:
            st.error(f"Optimization failed: {error}")

    optimization = st.session_state.get("optimization")
    if (
        isinstance(optimization, dict)
        and optimization["smiles"] == reference.strip()
        and optimization["objective"] == objective
    ):
        baseline = optimization["baseline"]
        analogs = optimization["analogs"]
        st.markdown("**Reference properties**")
        st.dataframe(pd.DataFrame([baseline]), hide_index=True, use_container_width=True)
        show_results(analogs, "eralpha_optimized_analogs")
