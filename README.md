---
title: ERalpha Molecule Studio
sdk: streamlit
app_file: app.py
---

# ERalpha Molecule Studio

Streamlit prototype for generating and exploring candidate ER-alpha ligands.
Install dependencies from `requirements.txt`; launch locally with `streamlit run app.py`.

For trained predictors, run `python train_models.py` to fetch ChEMBL206 activity data and
ClinTox toxicity labels, then serialize the models under `artifacts/`. Without those
artifacts the app uses small reference-only models, which are not suitable for research
or clinical decisions. ChemGPT weights are downloaded from Hugging Face when generation
is first requested.