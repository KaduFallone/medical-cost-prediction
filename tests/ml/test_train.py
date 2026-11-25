from ml.evaluate import evaluate
from ml.train import build_pipeline


def test_pipeline_predicts_one_value_per_row(sample_data):
    X, y = sample_data

    model = build_pipeline()
    model.fit(X, y)
    y_pred = model.predict(X)

    assert y_pred.shape == (len(X),)


def test_evaluate_returns_the_three_metrics(sample_data):
    X, y = sample_data

    model = build_pipeline().fit(X, y)
    metrics = evaluate(model, X, y)

    assert set(metrics) == {"mae", "rmse", "r2"}
