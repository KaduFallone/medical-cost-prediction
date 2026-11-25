import hashlib
import json
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.config import (
    CATEGORICAL_FEATURES,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    NUMERICAL_FEATURES,
    RAW_DATA_PATH,
    TARGET,
)
from ml.data import load_data, split_data
from ml.evaluate import evaluate
from ml.features import SmokerObeseFeature


def build_pipeline() -> Pipeline:
    """
    Building the model pipe:
    - add smoker_obese feature
    - preprocessing
    - training
    """
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(drop="first"), CATEGORICAL_FEATURES),
            ("int", "passthrough", ["smoker_obese"]),
        ]
    )

    return Pipeline(
        [
            ("features", SmokerObeseFeature()),
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def file_hash(file_path) -> str:
    """Return the SHA-256 hash of a file"""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def build_metadata(X_train, metrics: dict[str, float]) -> dict:
    """Return the model metadata saved next to the model"""

    return {
        "sklearn_version": sklearn.__version__,
        "features": list(X_train.columns),
        "target": TARGET,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "dataset_sha256": file_hash(RAW_DATA_PATH),
    }


def main() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    model = build_pipeline()
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    print(metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Model saved to {MODEL_PATH}")

    metadata = build_metadata(X_train, metrics)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"Metadata saved to {METADATA_PATH}")


if __name__ == "__main__":
    main()
