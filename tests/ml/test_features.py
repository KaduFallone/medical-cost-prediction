import pandas as pd

from ml.features import SmokerObeseFeature


def test_smoker_obese_flags_only_obese_smokers():
    X = pd.DataFrame(
        {
            "smoker": ["no", "no", "yes", "yes"],
            "bmi": [25.0, 35.0, 25.0, 30.0],
        }
    )

    result = SmokerObeseFeature().fit_transform(X)

    assert result["smoker_obese"].tolist() == [0, 0, 0, 1]


def test_smoker_obese_does_not_change_input():
    X = pd.DataFrame(
        {
            "smoker": ["yes"],
            "bmi": [35.0],
        }
    )

    SmokerObeseFeature().fit_transform(X)

    assert "smoker_obese" not in X.columns
