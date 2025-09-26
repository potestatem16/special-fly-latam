import fastapi
import os
import pandas as pd
from fastapi import HTTPException
from challenge.model import DelayModel

app = fastapi.FastAPI()

# Optional: load persisted estimator when requested
_model_instance = DelayModel()
if os.getenv("LOAD_MODEL", "false").lower() in {"1", "true", "yes"}:
    try:
        from joblib import load
        model_path = os.getenv("MODEL_PATH", "challenge/artifacts/model.joblib")
        _model_instance._model = load(model_path)
    except Exception:
        # Keep unfitted model if loading fails
        _model_instance._model = None

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(payload: dict) -> dict:
    # Minimal validation per tests
    if "flights" not in payload or not isinstance(payload["flights"], list):
        raise HTTPException(status_code=400)

    flights = payload["flights"]
    allowed_opera = {"Aerolineas Argentinas", "Grupo LATAM"}
    allowed_tipo = {"I", "N"}

    for f in flights:
        if not isinstance(f, dict):
            raise HTTPException(status_code=400)
        opera = f.get("OPERA")
        tipo = f.get("TIPOVUELO")
        mes = f.get("MES")
        if opera not in allowed_opera:
            raise HTTPException(status_code=400)
        if tipo not in allowed_tipo:
            raise HTTPException(status_code=400)
        if not isinstance(mes, int) or mes < 1 or mes > 12:
            raise HTTPException(status_code=400)

    df = pd.DataFrame(flights)
    features = _model_instance.preprocess(data=df)
    preds = _model_instance.predict(features)
    return {"predict": preds}