import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class EfficiencyTransformer(BaseEstimator, TransformerMixin):
    """Gets efficiency of columns."""

    def __init__(self, success_col: str, count_col: str):
        """
        Initializes with grouping column.

        Args:
            group_col (str): Grouping column.
        """
        self.success_col = success_col
        self.count_col = count_col

    def fit(self, X, y=None):
        """Dummy for inheritance."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Makes an efficiency column.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Dataframe with imputed values.
        """
        X_ = X.copy()
        X_[f"{self.success_col}_efficiency"] = X_.apply(
            lambda row: (
                row[self.success_col] / row[self.total_col]
                if row[self.total_col] != 0
                else 1
            ),
            axis=1,
        )
        return X_
