from fastapi import APIRouter

from app.model import explain, get_artifacts, predict
from app.schemas import ExplanationOut, PatientIn, PredictionOut

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint"""
    get_artifacts()
    return {"status": "ok"}


@router.get("/model")
def model_info() -> dict:
    """Return the model metadata"""
    _, metadata = get_artifacts()
    return metadata


@router.post("/predict")
def predict_charges(patient: PatientIn) -> PredictionOut:
    """Predict the medical cost (charges) for a single patient"""
    return PredictionOut(charges=predict(patient))


@router.post("/predict/explain")
def explanation(patient: PatientIn) -> ExplanationOut:
    """Show how much a feature contributed to the prediction for a single patient(adding, or removing values from the baseline)"""
    return ExplanationOut(**explain(patient))
