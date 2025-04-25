import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SpreadTransformer(BaseEstimator, TransformerMixin):
    """Adds point spread."""

    def fit(self, X, y=None):
        """Dummy for inheritance."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Expands the line_scores columns.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Dataframe with imputed values.
        """
        X_ = X.copy()
        X_["home_away_spread"] = X_["away_points"] - X_["home_points"]
        return X_
