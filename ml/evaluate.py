from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def evaluate(model, X, y) -> dict[str, float]:
    """Return the metrics"""

    y_pred = model.predict(X)

    return {
        "mae": float(mean_absolute_error(y, y_pred)),
        "rmse": float(root_mean_squared_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }
