import json
from functools import lru_cache

import joblib
import pandas as pd
import sklearn

from app.schemas import PatientIn
from ml.config import METADATA_PATH, MODEL_PATH


@lru_cache
def get_artifacts() -> tuple:
    """Load the model and metadata (fails if sklearn version has changed)"""
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())

    if metadata["sklearn_version"] != sklearn.__version__:
        raise RuntimeError(
            f"sklearn version has changed since the model was trained: "
            f"{metadata['sklearn_version']} != {sklearn.__version__}"
            f" Run `python -m ml.train` to retrain."
        )

    return model, metadata


def predict(patient: PatientIn) -> float:
    """Return the prediction for a single patient"""
    model, metadata = get_artifacts()

    X = pd.DataFrame([patient.model_dump()], columns=metadata["features"])

    return float(model.predict(X)[0])


def explain(patient: PatientIn) -> dict:
    """Split the prediction into a baseline plus one contribution per feature (linear model only)"""

    model, metadata = get_artifacts()

    X = pd.DataFrame([patient.model_dump()], columns=metadata["features"])

    X_transformed = model[:-1].transform(X)[0]
    regressor = model.named_steps["regressor"]
    names = model.named_steps["preprocessor"].get_feature_names_out()

    contributions = regressor.coef_ * X_transformed

    return {
        "baseline": float(regressor.intercept_),
        "contributions": dict(zip(names, contributions.tolist())),
        "charges": float(model.predict(X)[0]),
    }
