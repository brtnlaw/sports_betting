import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GroupMeanImputer(BaseEstimator, TransformerMixin):
    """Mean imputes the columns of a DataFrame grouped upon one column."""

    def __init__(self, group_col: str):
        """
        Initializes with grouping column.

        Args:
            group_col (str): Grouping column.
        """
        self.group_col = group_col

    def fit(self, X, y=None):
        """Dummy for inheritance."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Mean imputes all columns of a DataFrame according to one column.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Dataframe with imputed values.
        """
        X_ = X.copy()
        for col in X_.columns:
            if col != self.group_col:
                # Gets the mean by group
                X_.loc[(X_[col].isna()) & X_[self.group_col].notna(), col] = X_[
                    self.group_col
                ].map(X_.groupby(self.group_col)[col].mean())
                # Fills in the rest if not available
                X_[col] = X_[col].fillna(X_[col].mean())
        return X_
