---
title: ERalpha Molecule Studio
sdk: streamlit
app_file: app.py
---

# ERalpha Molecule Studio

Streamlit prototype for generating and exploring candidate molecules with predicted
ER-alpha activity and ClinTox toxicity probabilities.
Install dependencies from `requirements.txt`; launch locally with
`python3 -m streamlit run app.py`.

For trained predictors, run `python3 train_models.py` to fetch ChEMBL206 activity data and
ClinTox toxicity labels. The script evaluates the classifiers on held-out molecules and
saves the models plus `artifacts/metrics.json`. Without the model artifacts, the app uses
neutral placeholders that are not meaningful predictions. ChemGPT weights are downloaded
from Hugging Face when generation is first requested.

All displayed probabilities are computational screening estimates. They are not measured
binding values, evidence of ER-alpha agonism or antagonism, safety assessments, or clinical
evidence.
