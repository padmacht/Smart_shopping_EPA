# ShopSmart AI

[![CI](https://github.com/padmacht/Smart_shopping_EPA/actions/workflows/ci.yml/badge.svg)](https://github.com/padmacht/Smart_shopping_EPA/actions/workflows/ci.yml)

Customer segmentation, personalised recommendations and an AI shopping assistant (chatbot),
built with applied data science, machine learning, deep learning and MLOps through to deployment.

## Project structure

```
C:\Smart_shopping_EPA
├── data/
│   ├── raw/            ← the 8 ShopSmart datasets + DATA_CARD.md (never edit by hand)
│   ├── processed/      ← cleaned / feature tables created by code
│   └── external/       ← any extra data you bring in
├── configs/config.yaml ← dataset map (name → file) and project settings
├── src/shopsmart/      ← project code (importable package)
│   ├── config.py       ← paths + dataset map
│   ├── data/           ← load.py (load datasets by name), validate.py (data checks)
│   ├── segmentation/   ← features.py, model.py (K-Means), train.py
│   ├── recommender/    ← popularity, Item-KNN, hybrid, two-tower (deep learning) + evaluation
│   ├── search.py       ← catalogue search + Ollama query expansion
│   ├── search_eval.py  ← Hit@10 of search on the labelled queries
│   └── assistant.py    ← chat logic (intent, FAQ answers, price filters)
├── notebooks/          ← 01_data_overview.ipynb, then EDA and modelling notebooks
├── scripts/data_generation/  ← scripts that generated the synthetic datasets
├── tests/              ← pytest tests (run from the VS Code Testing panel)
├── models/             ← trained models (also logged to MLflow)
├── reports/figures/    ← charts for the report
├── api/                ← FastAPI inference service
├── app/                ← Streamlit app: app.py (entry, sidebar), common.py, views/ (4 pages)
├── .streamlit/config.toml ← green app theme
└── Smart_shopping_EPA.code-workspace ← VS Code settings, debug configs, tasks, extensions
```

## Datasets (data/raw)

| Name in code | File | Rows | What it is |
|---|---|---|---|
| `products` | products.csv | 1,500 | Product catalogue |
| `users` | user_data.csv | 5,000 | Customer profiles |
| `interactions` | interaction.csv | 354,340 | Views, clicks, carts, purchases, ratings in sessions |
| `reviews` | reviews.csv | 7,243 | Review text + star rating |
| `search_queries` | search_queries.csv | 515 | Labelled search queries |
| `chatbot_intents` | chatbot_intents.csv | 2,560 | Chatbot messages labelled with intents |
| `faq` | faq.csv | 16 | Shop policy Q&A |
| `personas_truth` | customer_personas_truth.csv | 5,000 | Hidden personas: **evaluation only** |

Full column descriptions: `data/raw/DATA_CARD.md`.

Load any dataset by name; no file paths needed:

```python
from shopsmart.data import load, load_all, list_datasets
products = load("products")
interactions = load("interactions")     # Timestamp already parsed as datetime
data = load_all()                        # everything except evaluation-only files
list_datasets()                          # summary table
```

## Setup in VS Code (Windows)

1. **Open the project.** In VS Code: *File → Open Workspace from File… → `C:\Smart_shopping_EPA\Smart_shopping_EPA.code-workspace`*
   (or double-click that file). This loads the project settings, debug configurations and tasks.
   Click **Install** when VS Code offers the recommended extensions.
   *Optional:* copy `.env.example` to `.env` and adjust the local Ollama settings for AI-powered product search.
2. **Create the environment.** *Terminal → Run Task… → "Setup: create venv + install requirements"*.
   Or in the terminal:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   pip install -e .
   ```
   If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
3. **Select the interpreter.** *Ctrl+Shift+P → "Python: Select Interpreter" → `.venv`*.
4. **Check the data.** *Terminal → Run Task… → "Validate data"* should end with all checks passed.
5. **Run the tests.** Open the Testing panel (beaker icon) and click ▶.
6. **Explore.** Open `notebooks/01_data_overview.ipynb`, choose the `.venv` kernel, and *Run All*.

**Viewing the CSVs:** click any file in `data/raw` to see coloured columns (Rainbow CSV), or right-click → *Open in Data Wrangler* for a spreadsheet view with filters and summaries.

## Train the models

```powershell
python -m shopsmart.segmentation.train              # K-Means customer segments (settings in config.yaml)
python -m shopsmart.segmentation.train --search-k   # elbow / silhouette table to choose k
```
Or in VS Code: *Terminal → Run Task… → "Train segmentation (K-Means)"*.
Outputs: `models/segmentation_kmeans.joblib`, `data/processed/customer_segments.csv`,
`reports/segmentation_profile.csv`, `reports/segmentation_metrics.json`.

## Train the recommenders

After generating customer segments, train the popularity baseline, Item-KNN, and hybrid recommender:

```powershell
python -m shopsmart.recommender.train
```

The training run uses a chronological 80/20 split, reports HitRate@10 and NDCG@10 for future
purchases, and saves the models to `models/recommenders.joblib`. Its metrics are written to
`reports/recommender_metrics.json`. For customers with no recorded interactions, recommendations
use their `customer_segments.csv` segment's training-period best-sellers; an unmapped customer
falls back to global best-sellers.

Load the saved models and recommend for a customer:

```python
from shopsmart.config import path
from shopsmart.recommender import RecommenderSuite

recommenders = RecommenderSuite.load(path("models") / "recommenders.joblib")
recommenders.recommend(user_id=123, model="hybrid", k=10)
# For an unseen customer, pass their assigned segment; otherwise global best-sellers are used.
recommenders.recommend(user_id=5001, model="hybrid", segment=3, k=10)
```

## Train the two-tower deep learning recommender

A two-tower neural network learns a 32-number embedding for every customer (ID + segment) and every
product (ID + category, type, brand and price). The score is how well the two embeddings match.
It uses the same time split and HitRate@10 / NDCG@10 test as the ML recommenders, so the
results are directly comparable. Train it after the ML recommenders:

```powershell
pip install torch                                   # once; only needed for training
python -m shopsmart.recommender.two_tower           # trains in under a minute on a laptop CPU
```
Or in VS Code: *Terminal → Run Task… → "Train two-tower (deep learning)"*.
Settings (epochs, temperature, embedding size) are under `two_tower:` in `configs/config.yaml`; they were
chosen on a validation period inside the training data, not on the test set.
Outputs: `models/two_tower.npz` (the learned embeddings as NumPy arrays, so the app needs no PyTorch)
and a `two_tower` entry in `reports/recommender_metrics.json`.

Result on the test period: the two-tower model reaches HitRate@10 13.5% (NDCG@10 0.050). That beats
best-sellers (12.4%) but not Hybrid (16.4%), so Hybrid stays the default recommender. Pick
**Two-tower (deep learning)** in the Shopping assistant sidebar to see its picks.

## Run the ShopSmart app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\app.py
```
Or in VS Code: *Terminal → Run Task… → "Run ShopSmart app"*, then open http://localhost:8501.
Run it from the project folder so the green theme in `.streamlit\config.toml` is applied.

| Page (sidebar) | Sidebar controls | What it shows |
|---|---|---|
| Overview | – | Data at a glance, the business problem, the three modules |
| Shopping assistant | Assistant mode: Search / Chat · Recommendation model | Search in your own words (results right below), chat, and "What should I buy next?" |
| Customer segments | View: All segments / One segment · Segmentation model | Segment sizes and profiles, campaign ideas, customer export |
| Model performance | Module: Recommendations / Segmentation / Search | Every model against its baseline |

The sidebar's **Demo: signed in as** picker chooses the customer for every page.
Segment display names and campaign ideas are in `configs/config.yaml` under `app:`;
update them if you retrain K-Means (segment numbers can change).

The app needs these generated outputs: `models/recommenders.joblib`,
`data/processed/customer_segments.csv`, `reports/*.json`. Rebuild them with the tasks
"Train segmentation (K-Means)", "Train recommenders" and "Evaluate search".

For local AI search, install [Ollama](https://ollama.com/download), run `ollama pull qwen2.5:1.5b`,
and start Ollama before the app. Without it, the app falls back to keyword search.

## MLOps, API and Docker

Training records parameters, evaluation metrics, and model/report artifacts in MLflow.
By default, the SQLite tracking database and artifacts are stored under `mlruns/`.
Set `MLFLOW_TRACKING_URI` to use an MLflow tracking server instead.

```powershell
# Train the models; each run is recorded in MLflow.
.\.venv\Scripts\python.exe -m shopsmart.segmentation.train
.\.venv\Scripts\python.exe -m shopsmart.recommender.train

# Browse local runs (in another terminal).
$mlflowDb = (Join-Path (Get-Location) "mlruns\mlflow.db").Replace("\", "/")
.\.venv\Scripts\mlflow.exe ui --backend-store-uri "sqlite:///$mlflowDb" --host 127.0.0.1 --port 5000
```

To log local training runs to the Dockerized MLflow server instead, start that service before
training and set the tracking URI in the training terminal:

```powershell
docker compose up --build -d mlflow
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
```

The FastAPI service uses the trained recommender and segment assignments. Start it locally:

```powershell
$env:PYTHONPATH = "src"
$env:SHOPSMART_ROOT = (Get-Location).Path
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Open `http://localhost:8000/docs`. The API includes:

| Route | Purpose |
|---|---|
| `GET /health` | Service health |
| `GET /recommendations/{user_id}?model=hybrid&k=10` | Recommend products; `model` can be `popularity`, `item_knn`, or `hybrid` |
| `GET /segments/{user_id}` | Return the customer's trained segment |
| `GET /search?query=headphones&use_llm=true` | Search the catalog, optionally expanding the query with local Ollama |

Build and run the Streamlit app, API, and MLflow tracking server in Docker:

```powershell
# Run after training so models/, data/processed/, and reports/ contain the outputs.
docker compose up --build
```

Services are available at Streamlit `http://localhost:8501`, API docs `http://localhost:8000/docs`,
and MLflow `http://localhost:5000`. Model/data/report directories are mounted read-only into
the application containers; MLflow state persists in the `mlflow_data` Docker volume.
Set `OLLAMA_BASE_URL` and `OLLAMA_MODEL` in the environment to override the local LLM defaults.

## CI/CD and deployment

**Live API:** https://smart-shopping-epa.onrender.com/docs (Render free tier: the first request after
15 idle minutes takes about a minute while it wakes up).

`.github/workflows/ci.yml` runs on GitHub Actions:

| Trigger | Job | What it does |
|---|---|---|
| Every push and pull request | **Test and quality gate** | Installs `requirements.txt`, validates the data, runs pytest, runs `scripts/check_model_quality.py` |
| After the tests pass | **Build and smoke-test Docker image** | Builds the image, starts the API inside it, calls `/health`, `/recommendations` and `/segments` |
| Mondays 06:17 UTC, or *Run workflow* in the Actions tab | **Retrain and evaluate models** | Retrains K-Means and the recommenders from the raw data, logs runs to MLflow, re-runs the quality gate, keeps `mlruns/`, reports and models as a downloadable artifact for 30 days |

The test job also runs `python -m shopsmart.search_eval`, which compares keyword search with
sentence-embedding search (`src/shopsmart/semantic_search.py`, Model2Vec `minishlab/potion-base-8M`,
NumPy only) and a hybrid of both on the 515 labelled queries; the table appears on the run's summary page.
The retrain job starts with an Evidently data drift report (`python -m shopsmart.drift`): the newest 20%
of events against the training period, per column, kept in the run's artifact as `reports/drift/`.
To run it locally: `pip install -r requirements-monitoring.txt` then `python -m shopsmart.drift`.

The quality gate fails the pipeline if Hybrid (the deployed recommender) does not beat the best-seller
baseline on HitRate@10, or if the segments are no better than chance (ARI <= 0).

Deployment: Render builds the `Dockerfile` from `main`. Set Render's *Settings → Auto-Deploy* to
**After CI Checks Pass** so a commit is only deployed once the pipeline above is green.

## Roadmap
| Week | Phase |
|---|---|
| 1 | ✅ Project setup, datasets mapped, validation · ⏭ EDA notebook, Git + DVC |
| 2 | Customer segmentation (RFM, K-Means/GMM, PCA) |
| 3 | Recommenders (popularity, Item-KNN, SVD, hybrid) + MLflow |
| 4 | Deep learning: autoencoder, two-tower recommender |
| 5 | Search: TF-IDF vs sentence embeddings + FAISS |
| 6 | Chatbot: intents, entities, Claude (optional) |
| 7 | FastAPI, Docker, tests, GitHub Actions, cloud deploy |
| 8 | Monitoring (Evidently), retraining, business value report |

> The data is synthetic (see DATA_CARD.md): results demonstrate methods, not real-world performance.
