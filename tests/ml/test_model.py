import json

import joblib
import pytest
import sklearn

from ml.config import METADATA_PATH, MODEL_PATH

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="run `python -m ml.train` first"
)


def test_saved_model_predicts_one_value_per_row(sample_data):
    X, y = sample_data
    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X)

    assert y_pred.shape == (len(X),)


def test_metadata_matches_environment_and_data(sample_data):
    X, y = sample_data
    metadata = json.loads(METADATA_PATH.read_text())

    assert metadata["sklearn_version"] == sklearn.__version__

    assert metadata["features"] == list(X.columns)
