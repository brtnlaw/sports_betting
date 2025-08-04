from typing import List, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NetTransformer(BaseEstimator, TransformerMixin):
    """Helper to generate the differences between a 'home' and 'away' column."""

    def __init__(
        self,
        home_cols: Union[str, List[str]],
        away_cols: Union[str, List[str]],
        new_col: str = None,
    ):
        """
        Sums up all the home_cols and all the away_cols, then nets.

        Args:
            home_cols (Union[str, List[str]]): Home columns to aggregate.
            away_cols (Union[str, List[str]]): Away columns to aggregate.
            new_col (str, optional): New column name. Defaults to "".
            drop_original (bool, optional) : Whether or not to drop the original columns. Defaults to True.
        """
        # or list...
        self.home_cols = [home_cols] if isinstance(home_cols, str) else home_cols
        self.away_cols = [away_cols] if isinstance(away_cols, str) else away_cols
        self.new_col = new_col

    def fit(self, X, y=None):
        """Dummy for inheritance."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generates net column to DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Dataframe with netted values.
        """
        X_ = X.copy()
        if self.new_col:
            col_name = self.new_col
        elif len(self.home_cols) == 1:
            if self.home_cols[0].startswith("home_"):
                root = self.home_col.split("home_")[1]
                col_name = f"net_{root}"
        elif len(self.away_cols) == 1:
            if self.away_col[0].startswith("away_")[1]:
                root = self.home_col.split("away_")[1]
                col_name = f"net_{root}"
        X_[col_name] = X_[self.home_cols].sum(axis=1) - X_[self.away_cols].sum(axis=1)
        X_.drop(columns=(self.home_cols + self.away_cols), axis=1, inplace=True)
        return X_
