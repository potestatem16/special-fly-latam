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
  - Run: `docker run -p 8080:8080 -e LOAD_MODEL=false delay-api`
  - Health: GET `http://localhost:8080/health`
- **Deployed API**: https://delay-api-f2yp47iw6q-uc.a.run.app

## Part IV - CI/CD Implementation with GitFlow and GitHub Actions

### GitFlow Workflow
- **main branch**: Production-ready code, triggers automatic deployment
- **dev branch**: Development branch for feature integration
- **Feature branches**: Created from dev for individual features

### CI/CD Pipeline

#### Continuous Integration (`.github/workflows/ci.yml`)
- **Triggers**: Push/PR to `main` branch
- **Environment**: Ubuntu latest with Python 3.11
- **Steps**:
  1. Checkout code
  2. Setup Python environment
  3. Install dependencies (`requirements-test.txt`, `requirements.txt`)
  4. Run model tests (`tests/model/test_model.py`)
  5. Run API tests (`tests/api/test_api.py`)
- **Status**: ✅ All tests passing

#### Continuous Deployment (`.github/workflows/cd.yml`)
- **Triggers**: Push to `main` branch only
- **Environment**: Ubuntu latest with GCP authentication
- **Steps**:
  1. Checkout code
  2. Authenticate with GCP using service account JSON
  3. Setup gcloud CLI
  4. Configure Docker for Artifact Registry
  5. Build Docker image with commit SHA tag
  6. Push image to Google Artifact Registry
  7. Deploy to Cloud Run with:
     - Port: 8080
     - Environment: `LOAD_MODEL=true`, `MODEL_PATH=/app/challenge/artifacts/model.joblib`
     - Public access enabled
- **Deployed URL**: https://delay-api-f2yp47iw6q-uc.a.run.app

### Configuration Files
- **`.coveragerc`**: Coverage configuration for test reports
- **`pytest.ini`**: Pytest settings (quiet output, Python path)
- **Required Secrets**:
  - `GCP_PROJECT`: Google Cloud Project ID
  - `GCP_SA_KEY`: Service Account JSON key for authentication
  - `SERVICE_NAME`: Cloud Run service name (default: delay-api)
  - `GCP_REGION`: Deployment region (default: us-central1)
  - `AR_REPO`: Artifact Registry repository (default: ml-api)

### Deployment Process
1. **Development**: Work on `dev` branch
2. **Testing**: CI runs automatically on PRs to `main`
3. **Merge**: Merge `dev` → `main` when ready
4. **Deploy**: CD automatically deploys to Cloud Run
5. **Verify**: API available at deployed URL
6. **Stress Test**: Update `STRESS_URL` in Makefile and run `make stress-test`


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
- Run API in Docker on port 8080:
  - `docker build -t delay-api .`
  - `docker run -p 8080:8080 -e LOAD_MODEL=false delay-api`
  - `Invoke-WebRequest http://127.0.0.1:8080/health | Select-Object -ExpandProperty Content`
- Stress test:
  - Local: `Invoke-WebRequest http://127.0.0.1:8080/health | Select-Object -ExpandProperty Content`
  - Deployed: `Invoke-WebRequest https://delay-api-f2yp47iw6q-uc.a.run.app/health | Select-Object -ExpandProperty Content`
  - `make stress-test` (STRESS_URL configured for deployed Cloud Run URL in Makefile)