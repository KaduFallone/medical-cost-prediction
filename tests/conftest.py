import pytest

from ml.config import RANDOM_STATE, TARGET
from ml.data import load_data


@pytest.fixture
def sample_data():
    """Small sample of the real dataset, so the tests run fast"""
    df = load_data().sample(n=60, random_state=RANDOM_STATE)

    X = df.drop(columns=TARGET)

    y = df[TARGET]

    return X, y
