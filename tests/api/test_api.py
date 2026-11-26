import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import predict
from app.schemas import PatientIn
from ml.config import MODEL_PATH

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="run `python -m ml.train` first"
)

VALID_PATIENT = {
    "age": 40,
    "sex": "male",
    "bmi": 32.0,
    "children": 1,
    "smoker": "yes",
    "region": "southeast",
}


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_returns_metadata(client):
    response = client.get("/model")

    assert response.status_code == 200
    assert "sklearn_version" in response.json()


def test_predict_returns_the_same_value_as_the_service(client):
    response = client.post("/predict", json=VALID_PATIENT)

    assert response.status_code == 200
    assert response.json()["charges"] == predict(PatientIn(**VALID_PATIENT))


def test_predict_rejects_age_out_of_range(client):
    patient = {**VALID_PATIENT, "age": 90}

    response = client.post("/predict", json=patient)

    assert response.status_code == 422


def test_explain_adds_up_to_the_prediction(client):
    response = client.post("/predict/explain", json=VALID_PATIENT)
    body = response.json()

    total = body["baseline"] + sum(body["contributions"].values())

    assert response.status_code == 200
    assert total == pytest.approx(body["charges"])
