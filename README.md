# Estrogen Receptor Blocker
About 70%-80% of breast cancers have estrogen receptors. Our platform is designed to generate, evaluate, and optimize novel drug candidates targeting the human estrogen receptor while minimizing toxicity.

## Methods
1. **Generate possible drugs** - most drugs that inhibit the breast cancer target share a core chemical backbone called a triphenylethylene core or a phenol group. A partial molecule input along with some configurations are passed to [ChemGPT-4.7M](https://huggingface.co/ncfrey/ChemGPT-4.7M) to generate possible molecules.
2. **Optimize existing drugs** - a complete molecule can be slightly transformed using [rdChemReactions](https://www.rdkit.org/docs/source/rdkit.Chem.rdChemReactions.html) based on our [list](./optimizer.py) of possible transformations to attempt to get better measurements.

Every molecule generated/optimized are passed into [RDKit.Chem](https://www.rdkit.org/) to analyze the molecular weight, LogP, HBD/HBA, and synthetic accessibility. Only molecules that pass the filters for these metrics are shown as results.

The toxicity and ER-alpha activity values are predicted using [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) from these datasets:
1. [ClinTox](https://huggingface.co/datasets/zpn/clintox) includes drugs approved by the FDA and those that have failed clinical trials for toxicity reasons. We use this dataset to predict the toxicity value.
2. [CHEMBL206](https://www.ebi.ac.uk/chembl/explore/target/CHEMBL206) includes bioacticity records of molecules that interact with the Estrogen receptor alpha. We use this dataset to predict the ERalpha activity value.

The training information is available in [metrics.json](./artifacts/metrics.json) or using the "View model training info" button on webpage.

## Tech Stack
The client web UI (**HTML/CSS/JS**) sends HTTPS **REST** requests to a Python **Rust**/**FastAPI** server. The backend is fully written in Python, using `rdkit`, `scikit-learn`, and other libraries.


## Running the Project
Install the Python dependencies"
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

If you want to train the model:
```bash
rm ./artifacts/*
python3 train_models.py
```

If you want to use our pre-trained artifacts, they are already included in [artifacts](./artifacts/), no action needed.

Start the application:
```bash
python3 -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000>.
API doc is at <http://localhost:8000/docs>.

## Environment Variables
In the `.env` file (not included in this repository), these values are required:
- `HF_TOKEN`, a [Hugging Face](https://huggingface.co/) access token
