import os
import sys
import pandas as pd
from joblib import dump

# Ensure project root is on sys.path when executed as `python scripts/train.py`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from challenge.model import DelayModel


def main() -> None:
    required_columns = ["Fecha-I", "Fecha-O", "OPERA", "TIPOVUELO", "MES"]
    df = pd.read_csv(
        "data/data.csv",
        usecols=required_columns,
        low_memory=False,
    )
    model = DelayModel()
    X, y = model.preprocess(df, target_column="delay")
    model.fit(X, y)
    dump(model._model, "artifacts/model.joblib")


if __name__ == "__main__":
    main()
