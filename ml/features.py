from sklearn.base import BaseEstimator, TransformerMixin


class SmokerObeseFeature(BaseEstimator, TransformerMixin):
    """
    Add a column indicating whether a patient is both a
    smoker and obese (1 if true , 0 if false).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["smoker_obese"] = ((X["smoker"] == "yes") & (X["bmi"] >= 30)).astype(int)

        return X
