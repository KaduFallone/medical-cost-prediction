from typing import Literal

from pydantic import BaseModel, Field


class PatientIn(BaseModel):
    """Input of /predict: one person's data (same columns as the training data)"""

    age: int = Field(ge=18, le=64)
    sex: Literal["male", "female"]
    bmi: float = Field(ge=15, lt=55)
    children: int = Field(ge=0, le=5)
    smoker: Literal["yes", "no"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]


class PredictionOut(BaseModel):
    """Output of /predict: the predicted medical cost (charges)"""

    charges: float


class ExplanationOut(BaseModel):
    """Output of /explain: the predicted medical cost (charges) plus a breakdown of contributions"""

    baseline: float
    contributions: dict[str, float]
    charges: float
