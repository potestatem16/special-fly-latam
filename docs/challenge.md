## Part I - Model operationalization notes

### Implementation Summary
- Transcribed DS notebook into `challenge/model.py` with `preprocess`, `fit`, `predict` methods
- Fixed typing bug: `Union(...)` → `Union[...]` (common notebook-to-script conversion error)
- All 4 model tests pass

### Key Features
- **Fixed 10-feature schema** (required by tests): `OPERA_Latin American Wings`, `MES_7`, `MES_10`, `OPERA_Grupo LATAM`, `MES_12`, `TIPOVUELO_I`, `MES_4`, `MES_11`, `OPERA_Sky Airline`, `OPERA_Copa Air`
- **Delay computation**: `min_diff = (Fecha-O − Fecha-I) [minutes]`, `delay = 1 if min_diff > 15 else 0`
- **LogisticRegression** with class weights to handle imbalance: `class_weight={1: n_y0/len(y), 0: n_y1/len(y)}`
- **Safe fallback**: Returns zeros if model not fitted

### Model Choice Rationale
- Logistic Regression chosen over XGBoost for simplicity, speed, determinism
- Comparable recall behavior to XGBoost with `scale_pos_weight`
- Vectorized preprocessing, no row-wise loops


## Part II - API operationalization notes

- Endpoints implemented in `challenge/api.py` (exposed as `challenge.app`):
  - `GET /health` → `{"status":"OK"}`.
  - `POST /predict` expects body: `{"flights": [{"OPERA": str, "TIPOVUELO": "I"|"N", "MES": 1..12}, ...]}`.
- Minimal validation (per tests):
  - `OPERA` must be in allowlist: `{ "Aerolineas Argentinas", "Grupo LATAM" }`.
  - `TIPOVUELO` ∈ {`I`, `N`}.
  - `MES` integer in [1..12].
  - Any violation → HTTP 400.
- Serving behavior:
  - Builds a DataFrame from `flights`, calls `DelayModel.preprocess` and `DelayModel.predict`.
  - No training at startup; unfitted model returns zeros, matching `tests/api/test_api.py` expectations.
- How to run API tests:
  - `cd tests`
  - `python -m pytest -q api/test_api.py`

## Part III - Train artifact and Docker notes

- Artifacts:
  - Trained estimator saved as `challenge/artifacts/model.joblib`.
- Training (from repo root):
  - `pip install -r requirements.txt`
  - `python .\challenge\train.py` (PowerShell) or `python ./challenge/train.py`
  - CSV read optimized with `usecols=["Fecha-I","Fecha-O","OPERA","TIPOVUELO","MES"]` and `low_memory=False`.

- Runtime toggle:
  - `LOAD_MODEL=true` and optional `MODEL_PATH=/app/challenge/artifacts/model.joblib`.
- Docker (prod):
  - Build: `docker build -t delay-api .`
  - Run: `docker run -p 8000:8000 -e LOAD_MODEL=false delay-api`
  - Health: GET `http://localhost:8000/health`

## Part IV - CI/CD notes

- Added `.coveragerc` at the repo root to match the `--cov-config=../.coveragerc` argument used by the Makefile and pytest commands. 
- `pytest.ini` remains at the repo root to ensure pytest runs from the root directory, sets `pythonpath=.` and uses `-q` for quiet output.
- CI at `.github/workflows/ci.yml` runs model and API tests on push/PR to `main`.
- CD at `.github/workflows/cd.yml` builds and deploys to Cloud Run. Requires secrets:
  - `GCP_PROJECT`, `GCP_SA_KEY`.
- After deploy, update `Makefile` `STRESS_URL` with the service URL and run `make stress-test`.


## Windows PowerShell run steps

- Activate venv (already created):
  - `& .\venv\Scripts\Activate.ps1`
- Install deps via Makefile:
  - `make install`
- Run tests via Makefile (targets `cd` into `tests` and write reports to `reports/`):
  - `make model-test`
  - `make api-test`
- Train and persist model:
  - `python .\challenge\train.py`
- Run API in Docker on port 8000:
  - `docker build -t delay-api .`
  - `docker run -p 8000:8000 -e LOAD_MODEL=false delay-api`
  - `Invoke-WebRequest http://127.0.0.1:8000/health | Select-Object -ExpandProperty Content`
- Stress test (update `STRESS_URL` in `Makefile` when deployed):
  - `make stress-test`