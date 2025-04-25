import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GroupModeImputer(BaseEstimator, TransformerMixin):
    """Mode imputes the columns of a DataFrame grouped upon one column."""

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
        Mode imputes all columns of a DataFrame according to one column.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Dataframe with imputed values.
        """
        X_ = X.copy()
        for col in X_.columns:
            if col != self.group_col:
                mode_df = (
                    X_.groupby(self.group_col)[col]
                    .apply(lambda group: group.mode())
                    .reset_index(level=0)
                )
                mode_dict = dict(zip(mode_df[self.group_col], mode_df[col]))
                X_[col] = X_.apply(
                    lambda row: mode_dict.get(row[self.group_col], row[col]), axis=1
                )
        return X_
