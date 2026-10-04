# Estrogen Receptor Blocker


## Run locally

Install the Python dependencies and either train or copy the model artifacts:
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 train_models.py
```

Start the application:
```bash
source .venv/bin/activate
python3 -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000>.
API doc is at <http://localhost:8000/docs>.


## Models

`train_models.py` fetches ChEMBL206 activity data and ClinTox toxicity labels, evaluates
the classifiers, and writes `artifacts/affinity.joblib`,
`artifacts/toxicity.joblib`, and `artifacts/metrics.json`.

ChemGPT weights are downloaded from Hugging Face when generation is first requested.

